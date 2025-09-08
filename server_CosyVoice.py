import uvicorn
import logging
import time
import base64
import wave
import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
from cloneForTTS import CloneForTTS
from cosyVoiceTTS import CosyVoiceTTS
import numpy as np
from fastapi.responses import StreamingResponse, Response
from fastapi.middleware.cors import CORSMiddleware


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

app = FastAPI(title="CosyVoice TTS API", version="1.0.0")

# CORS中间件配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源
    allow_credentials=True,
    allow_methods=["*"],  # 允许所有HTTP方法
    allow_headers=["*"],  # 允许所有请求头
)

def apply_volume_adjustment(speech_data, volume):
    """应用智能音量调整修复音量过低问题"""
    # 确保音量在合理范围内
    volume = max(0.1, min(10.0, volume))
    
    # 检查原始音频的动态范围
    max_val = np.abs(speech_data).max()
    
    if max_val == 0:
        # 如果是静音，直接返回
        logger.warning("检测到静音音频数据")
        return np.zeros_like(speech_data, dtype=np.int16)
    
    # 智能音量调整：确保音频有足够的电平
    # CosyVoice2输出的原始信号通常比较小，需要适当放大
    if max_val < 0.1:  # 如果原始信号很小
        # 先进行基础放大，使信号达到合理水平
        base_gain = 0.8 / max_val  # 放大到80%满量程
        speech_data = speech_data * base_gain
        max_val = 0.8
        logger.info(f"原始信号过小，先进行基础放大: {base_gain:.2f}倍")
    
    # 应用用户指定的音量
    target_max = 0.8 * volume  # 目标最大值，留20%余量防止削峰
    if target_max > 0.95:  # 限制最大值，防止严重削峰
        target_max = 0.95
    
    # 计算最终的音量乘数
    volume_multiplier = target_max / max_val
    speech_data = speech_data * volume_multiplier
    # 转换为16位整数，应用最终的量化
    final_multiplier = 32767
    result = (speech_data * final_multiplier).astype(np.int16)
    
    # 统计输出信息
    final_max = np.abs(result).max()
    final_rms = np.sqrt(np.mean(result.astype(np.float64) ** 2))
    
    logger.info(f"音量调整完成: 用户音量={volume:.1f}, 最大振幅={final_max}({final_max/32767*100:.1f}%), RMS={final_rms:.0f}")
    
    return result

def save_audio_as_wav(audio_data_chunks, filename="test.wav", sample_rate=24000):
    """将音频数据保存为WAV文件"""
    try:
        # 确保输出目录存在
        output_dir = "outputs"
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"创建输出目录: {output_dir}")
        
        filepath = os.path.join(output_dir, filename)
        
        # 合并所有音频块
        all_audio_data = b''.join(audio_data_chunks)
        
        # 转换为numpy数组
        audio_array = np.frombuffer(all_audio_data, dtype=np.int16)
        
        # 保存为WAV文件
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # 单声道
            wav_file.setsampwidth(2)  # 16位 = 2字节
            wav_file.setframerate(sample_rate)  # 采样率
            wav_file.writeframes(all_audio_data)
        
        file_size = os.path.getsize(filepath)
        duration = len(audio_array) / sample_rate
        
        logger.info(f"[成功] 音频文件保存成功:")
        logger.info(f"   - 文件路径: {filepath}")
        logger.info(f"   - 文件大小: {file_size/1024:.1f} KB")
        logger.info(f"   - 音频时长: {duration:.2f} 秒")
        logger.info(f"   - 采样率: {sample_rate} Hz")
        logger.info(f"   - 声道数: 1 (单声道)")
        logger.info(f"   - 位深度: 16 bit")
        
        return filepath, file_size, duration
        
    except Exception as e:
        logger.error(f"[错误] 保存音频文件失败: {e}")
        raise e

# 定义一个数据模型，用于接收POST请求中的数据
class TTSRequest(BaseModel):
    tts_text: Optional[str] = None      # 待合成的文本
    instruct_text: Optional[str] = None  # 指令文本
    stream: Optional[bool] = False      # 是否使用流式合成
    speed: Optional[float] = 1.0        # 语速
    volume: Optional[float] = 2.0       # 音量（提高默认值以获得更好的音频电平）
    save_file: Optional[bool] = False   # 是否保存为文件

class CloneForTTSRequest(BaseModel):
    audio_data_b64: Optional[str] = None  # 音频数据

# 初始化TTS引擎
logger.info("正在初始化CosyVoice TTS引擎...")
try:
    cosyvoice = CosyVoiceTTS()
    cosyvoice.register_audio_data()
    logger.info("[成功] CosyVoice TTS引擎初始化成功")
except Exception as e:
    logger.error(f"[错误] CosyVoice TTS引擎初始化失败: {e}")
    raise e
@app.get("/health")
async def health_check():
    """健康检查端点"""
    try:
        # 检查TTS引擎状态
        engine_status = "ok" if cosyvoice else "error"
        return {
            "status": "ok",
            "message": "TTS服务运行正常",
            "engine_status": engine_status,
            "timestamp": time.time()
        }
    except Exception as e:
        logger.error(f"健康检查失败: {e}")
        raise HTTPException(status_code=500, detail="TTS服务异常")

@app.post("/cosyvoice")
async def tts(request: TTSRequest):
    """CosyVoice TTS合成端点"""
    start_time = time.time()
    
    # 参数验证
    if not request.tts_text or not request.tts_text.strip():
        raise HTTPException(status_code=400, detail="tts_text不能为空")
    
    if len(request.tts_text) > 1000:
        raise HTTPException(status_code=400, detail="文本长度不能超过1000字符")
    
    if not (0.5 <= request.speed <= 2.0):
        raise HTTPException(status_code=400, detail="语速必须在0.5-2.0之间")
    
    # 记录请求信息
    logger.info(f"[请求] TTS请求 - 文本长度: {len(request.tts_text)}, 流式: {request.stream}, 语速: {request.speed}")
    logger.info(f"[内容] 文本内容: {request.tts_text[:50]}{'...' if len(request.tts_text) > 50 else ''}")
    
    # 设置响应头 - 使用优化的音频格式配置
    headers = {
        "Content-Type": "audio/pcm",
        "X-Sample-Rate": "24000",           # CosyVoice2标准输出采样率
        "X-Channel-Count": "1",             # 单声道
        "X-Bits-Per-Sample": "16",          # 16位PCM
        "X-Audio-Format": "PCM-LE",         # 小端序PCM格式
        "X-Buffer-Size": "24576",           # 优化的缓冲区大小
        "Access-Control-Expose-Headers": "Content-Type,X-Sample-Rate,X-Channel-Count,X-Bits-Per-Sample,X-Audio-Format,X-Buffer-Size"
    }
    
    # 如果是流式请求，添加流式标识
    if request.stream:
        headers.update({
            "X-Stream": "true",
            "Transfer-Encoding": "chunked",
            "Cache-Control": "no-cache"
        })
        logger.info("[流式] 启用流式传输模式")
    
    async def generate_audio():
        """音频生成器"""
        chunk_count = 0
        total_bytes = 0
        audio_chunks = []  # 用于保存文件时收集所有音频块
        
        try:
            if request.stream:
                # 流式模式
                logger.info("[生成] 开始流式音频生成...")
                voice_response = cosyvoice.generate_audio_stream(
                    tts_text=request.tts_text,
                    instruct_text=request.instruct_text or "请用自然流畅的语调朗读",
                    volume=request.volume,
                    speed=request.speed
                )
                
                for audio_data in voice_response:
                    chunk_count += 1
                    # generate_audio_stream返回的数据格式：{'data': base64_string, 'chunk_id': int, 'size': int, ...}
                    if 'data' in audio_data:
                        # 音频数据是base64编码的，需要解码为bytes
                        chunk_bytes = base64.b64decode(audio_data['data'])
                        
                        # 记录详细信息（只对前几个块）
                        if chunk_count <= 3:
                            logger.info(f"[数据] 流式块 {chunk_count}: 解码 {len(audio_data['data'])} 字符base64 → {len(chunk_bytes)} 字节PCM")
                        elif chunk_count % 10 == 0:
                            logger.info(f"[数据] 流式块 {chunk_count}: {len(chunk_bytes)} 字节")
                    else:
                        logger.warning(f"[警告] 未知的音频数据格式，期望包含'data'字段: {audio_data.keys()}")
                        continue
                    
                    total_bytes += len(chunk_bytes)
                    
                    # 如果需要保存文件，收集音频块
                    if request.save_file:
                        audio_chunks.append(chunk_bytes)
                    
                    if chunk_count <= 3:  # 只记录前几个块避免日志过多
                        logger.info(f"[数据] 流式块 {chunk_count}: {len(chunk_bytes)} 字节")
                    elif chunk_count % 10 == 0:  # 每10块记录一次
                        logger.info(f"[数据] 流式块 {chunk_count}: {len(chunk_bytes)} 字节 (总计: {total_bytes/1024:.1f}KB)")
                    
                    yield chunk_bytes
                    
            else:
                # 非流式模式
                logger.info("[生成] 开始非流式音频生成...")
                audio_generator = cosyvoice.generate_audio(
                    text=request.tts_text,
                    instruct_text=request.instruct_text or "请用自然流畅的语调朗读",
                    stream=False,
                    speed=request.speed
                )
                
                for out in audio_generator:
                    chunk_count += 1
                    # 修复：使用音量调整而不是硬编码32767
                    speech_data = out['tts_speech'].numpy()
                    raw = apply_volume_adjustment(speech_data, request.volume)
                    chunk_bytes = raw.tobytes()
                    total_bytes += len(chunk_bytes)
                    
                    # 如果需要保存文件，收集音频块
                    if request.save_file:
                        audio_chunks.append(chunk_bytes)
                    
                    logger.info(f"[数据] 音频块 {chunk_count}: {len(chunk_bytes)} 字节")
                    yield chunk_bytes
            
            # 生成完成
            elapsed_time = time.time() - start_time
            logger.info(f"[完成] 音频生成完成 - 块数: {chunk_count}, 总大小: {total_bytes/1024:.1f}KB, 耗时: {elapsed_time:.2f}s")
            
            # 如果需要保存文件
            if request.save_file and audio_chunks:
                try:
                    filepath, file_size, duration = save_audio_as_wav(audio_chunks, "test.wav")
                    logger.info(f"[保存] 音频已保存到: {filepath}")
                except Exception as save_error:
                    logger.error(f"[错误] 保存音频文件时出错: {save_error}")
            
        except Exception as e:
            logger.error(f"[错误] 音频生成失败: {e}")
            # 在流式响应中，我们不能抛出HTTP异常，只能记录错误
            error_msg = f"TTS生成错误: {str(e)}"
            yield error_msg.encode('utf-8')
    
    return StreamingResponse(
        generate_audio(),
        media_type="audio/pcm",
        headers=headers
    )

@app.post("/cloneForTTS")
async def cloneForTTS(request: CloneForTTSRequest):
    """克隆TTS合成端点，用于克隆音频数据"""
    cloneForTTS = CloneForTTS()
    audio_tensor = cloneForTTS.decode_audio_data(request.audio_data_b64)
    cosyvoice.set_recorded_audio(audio_tensor)
    return {"message": "音频数据设置成功"}

@app.get("/api/info")
async def api_info():
    """API信息端点"""
    return {
        "name": "CosyVoice TTS API",
        "version": "1.0.0",
        "description": "基于CosyVoice的文字转语音服务",
        "endpoints": {
            "/health": "健康检查",
            "/cosyvoice": "TTS合成 (支持流式传输)",
            "/api/info": "API信息"
        },
        "features": [
            "流式音频生成",
            "可调节语速",
            "自定义指令文本",
            "PCM格式输出"
        ],
        "supported_formats": {
            "output": "16-bit PCM",
            "sample_rate": "24000 Hz",
            "channels": "1 (Mono)"
        }
    }

if __name__ == '__main__':
    logger.info("[启动] 正在启动CosyVoice TTS服务器...")
    logger.info("[配置] 服务配置:")
    logger.info("   - 主机: 0.0.0.0")
    logger.info("   - 端口: 8800")
    logger.info("   - 支持流式传输: [OK]")
    logger.info("   - 支持CORS: [OK]")
    logger.info("   - 音频格式: 16-bit PCM, 24kHz, Mono")
    logger.info("[API] API端点:")
    logger.info("   - GET  /health     - 健康检查")
    logger.info("   - POST /cosyvoice  - TTS合成")
    logger.info("   - GET  /api/info   - API信息")
    logger.info("[提示] 前端测试地址: http://localhost:8800/docs")
    
    try:
        uvicorn.run(
            app, 
            host='0.0.0.0', 
            port=8800,
            log_level="info",
            access_log=True
        )
    except KeyboardInterrupt:
        logger.info("[停止] 服务器已停止")
    except Exception as e:
        logger.error(f"[错误] 服务器启动失败: {e}")
        raise e