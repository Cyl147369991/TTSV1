import logging
import time
import numpy as np
import torch
import base64
import traceback
import sys
import os

# 添加项目根目录到路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.append(project_root)
from config.model_config import get_model_cache_dir

# 获取CosyVoice路径
COSYVOICE_PATH = os.path.join(os.path.dirname(__file__), 'CosyVoice')
sys.path.insert(0, "D:/upmicProject/BU2-Sys/AIAssistant/TTS/CosyVoice/third_party/Matcha-TTS")
def setup_cosyvoice():
    """手动设置CosyVoice环境"""
    print(f"🔍 当前工作目录: {os.getcwd()}")
    print(f"🔍 当前文件路径: {__file__}")
    print(f"🔍 CosyVoice路径: {COSYVOICE_PATH}")
    print(f"🔍 路径是否存在: {os.path.exists(COSYVOICE_PATH)}")
    
    # 强制重新设置路径（确保优先级）
    if COSYVOICE_PATH in sys.path:
        sys.path.remove(COSYVOICE_PATH)
    sys.path.insert(0, COSYVOICE_PATH)
    print(f"🔍 已重新添加到sys.path: {COSYVOICE_PATH}")
    
    # 同时添加一些可能需要的额外路径
    extra_paths = [
        os.path.join(COSYVOICE_PATH, 'third_party', 'Matcha-TTS'),
        os.path.dirname(__file__)  # TTS目录
    ]
    
    for extra_path in extra_paths:
        if os.path.exists(extra_path) and extra_path not in sys.path:
            sys.path.insert(0, extra_path)
            print(f"🔍 已添加额外路径: {extra_path}")
    
    # 检查必要的文件
    required_files = [
        'cosyvoice',
        'pretrained_models'  # 如果有的话
    ]
    
    for file in required_files:
        full_path = os.path.join(COSYVOICE_PATH, file)
        if os.path.exists(full_path):
            print(f"✅ 找到: {file}")
        else:
            print(f"❌ 缺失: {file}")
    
    # 检查cosyvoice模块的具体路径
    cosyvoice_module_path = os.path.join(COSYVOICE_PATH, 'cosyvoice')
    cosyvoice_cli_path = os.path.join(cosyvoice_module_path, 'cli')
    cosyvoice_py_path = os.path.join(cosyvoice_cli_path, 'cosyvoice.py')
    print(f"🔍 cosyvoice模块路径: {cosyvoice_module_path}")
    print(f"🔍 cosyvoice.cli路径: {cosyvoice_cli_path}")
    print(f"🔍 cosyvoice.py文件: {cosyvoice_py_path}")
    print(f"🔍 cosyvoice.py存在: {os.path.exists(cosyvoice_py_path)}")
    
    return True

setup_cosyvoice()

# 推迟导入CosyVoice2，避免初始化时的导入错误
CosyVoice2 = None
load_wav = None

def _import_cosyvoice():
    """延迟导入CosyVoice2模块"""
    global CosyVoice2, load_wav
    if CosyVoice2 is None:
        # 确保路径设置正确
        setup_cosyvoice()
        
        try:
            from cosyvoice.cli.cosyvoice import CosyVoice2 as _CosyVoice2
            from cosyvoice.utils.file_utils import load_wav as _load_wav
            CosyVoice2 = _CosyVoice2
            load_wav = _load_wav
            print("✅ 成功导入 CosyVoice2 模块")
        except ImportError as e:
            print(f"❌ 导入失败: {e}")
            print("尝试强制重置Python路径缓存...")
            
            # 清除模块缓存
            import importlib
            if 'cosyvoice' in sys.modules:
                del sys.modules['cosyvoice']
            if 'cosyvoice.cli' in sys.modules:
                del sys.modules['cosyvoice.cli']
            if 'cosyvoice.cli.cosyvoice' in sys.modules:
                del sys.modules['cosyvoice.cli.cosyvoice']
            
            # 重新强制设置路径
            extra_path = os.path.join(os.path.dirname(__file__), 'CosyVoice')
            if extra_path in sys.path:
                sys.path.remove(extra_path)
            sys.path.insert(0, extra_path)
            
            try:
                from cosyvoice.cli.cosyvoice import CosyVoice2 as _CosyVoice2
                from cosyvoice.utils.file_utils import load_wav as _load_wav
                CosyVoice2 = _CosyVoice2
                load_wav = _load_wav
                print("✅ 清除缓存后成功导入")
            except ImportError as e2:
                print(f"❌ 仍然导入失败: {e2}")
                print(f"当前sys.path: {sys.path[:5]}...")  # 只显示前5个路径
                raise
    return CosyVoice2, load_wav

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('assistant.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# 音频缓冲配置 - 优化跨平台兼容性和音质
BUFFER_SIZE = 24576  # 24KB缓冲区，提高音质稳定性
MIN_CHUNK_SIZE = 6144   # 6KB最小块，保证音频连续性
MAX_BUFFER_TIME = 0.08  # 80ms最大缓冲时间，平衡延迟和稳定性

# 流式播放优化配置 - 针对网络传输优化
STREAM_BUFFER_SIZE = 32768   # 32KB流式缓冲区，减少网络抖动影响
STREAM_MIN_CHUNK_SIZE = 12288 # 12KB最小流式块，确保音频质量
STREAM_PREFETCH_TIME = 0.15   # 150ms预取时间，优化播放流畅度

# 采样率配置 - 确保跨平台一致性
OUTPUT_SAMPLE_RATE = 24000    # CosyVoice2标准输出采样率
INPUT_SAMPLE_RATE = 16000     # 输入音频标准采样率
AUDIO_CHANNELS = 1            # 单声道输出
AUDIO_BITS_PER_SAMPLE = 16    # 16位PCM

# 初始化CosyVoice模型
model_path = os.path.join(get_model_cache_dir("modelscope"), "iic\\CosyVoice2-0___5B")

# 优化模型加载和配置
def optimize_cosyvoice_model():
    # 导入CosyVoice2模块
    CosyVoice2_class, _ = _import_cosyvoice()
    
    # 启用优化设置
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
    # 加载模型时的优化
    cosyvoice = CosyVoice2_class(
        model_path,
        load_jit=True,  # 禁用JIT编译以避免兼容性问题
        load_trt=True,  # 禁用TensorRT以避免版本不兼容问题
        fp16=True       # 禁用fp16以确保兼容性
    )
    return cosyvoice


class AudioBuffer:
    """音频缓冲类，用于优化流式传输"""
    def __init__(self, buffer_size=BUFFER_SIZE, min_chunk_size=MIN_CHUNK_SIZE, stream_mode=False):
        # 根据是否为流式模式选择不同的缓冲区配置
        if stream_mode:
            self.buffer_size = STREAM_BUFFER_SIZE
            self.min_chunk_size = STREAM_MIN_CHUNK_SIZE
            self.max_buffer_time = MAX_BUFFER_TIME
        else:
            self.buffer_size = buffer_size
            self.min_chunk_size = min_chunk_size
            self.max_buffer_time = MAX_BUFFER_TIME
            
        self.buffer = bytearray()
        self.last_yield_time = time.time()
        self.stream_mode = stream_mode
        self.chunk_count = 0
        
    def add_data(self, data):
        """添加数据到缓冲区"""
        self.buffer.extend(data)
        
    def should_yield(self):
        """判断是否应该输出数据"""
        current_time = time.time()
        time_since_last_yield = current_time - self.last_yield_time
        
        if self.stream_mode:
            # 流式模式：更智能的输出策略
            # 1. 达到最小块大小就可以输出（保证低延迟）
            # 2. 或者达到缓冲区大小（保证效率）
            # 3. 或者等待时间过长（避免卡顿）
            return (len(self.buffer) >= self.min_chunk_size or 
                    len(self.buffer) >= self.buffer_size or
                    (len(self.buffer) > 0 and time_since_last_yield > self.max_buffer_time))
        else:
            # 普通模式：原有逻辑
            return (len(self.buffer) >= self.buffer_size or 
                    (len(self.buffer) >= self.min_chunk_size and time_since_last_yield > self.max_buffer_time))
    
    def get_data(self):
        """获取缓冲区数据并清空"""
        if len(self.buffer) > 0:
            data = bytes(self.buffer)
            self.buffer.clear()
            self.last_yield_time = time.time()
            self.chunk_count += 1
            return data
        return None
    
    def flush(self):
        """强制输出剩余数据"""
        return self.get_data()
    
    def get_stats(self):
        """获取缓冲区统计信息"""
        return {
            'buffer_size': len(self.buffer),
            'chunk_count': self.chunk_count,
            'stream_mode': self.stream_mode
        }


class CosyVoiceTTS:
    def __init__(self,tts_volume=1.0,tts_speed=1.0,video_path="input.wav"):
        self.recorded_audio = None  # 存储录制的音频
        self.default_audio = None   # 存储默认音频
        self.prompt_speech_16k = None
        self.cosyvoice = optimize_cosyvoice_model()
        self.load_default_audio(video_path if video_path is not None else "input.wav")
        self._is_registered = False  # 添加标志位
        self.stop_inference = False  # 添加推理停止标志
        pass

    def load_default_audio(self,file_path="input.wav"):
        """加载默认音频"""
        try:
            self.default_audio = load_default_audio(file_path)
            self.prompt_speech_16k = self.default_audio
            logger.info("默认音频加载完成")
        except Exception as e:
            logger.error(f"加载默认音频失败: {str(e)}")
            self.default_audio = prepare_audio_data(None, target_length=16000)
    
    def get_prompt_audio(self):
        """获取用于TTS的prompt音频（优先使用录制音频，否则使用默认音频）"""
        if self.recorded_audio is not None:
            logger.info("使用录制的音频作为prompt")
            return self.recorded_audio
        else:
            logger.info("使用默认音频作为prompt")
            return self.default_audio

    def request_stop_inference(self):
        """请求停止推理过程"""
        logger.info("收到停止推理请求")
        self.stop_inference = True

    def reset_stop_flag(self):
        """重置停止标志"""
        self.stop_inference = False

    def apply_volume_adjustment(self,speech_data, volume):
        """应用音量调整"""
        # 确保音量在合理范围内
        volume = max(0.1, min(7.0, volume))
        
        # 计算调整后的乘数，避免音频削峰
        base_multiplier = 32767
        adjusted_multiplier = base_multiplier * volume
        
        # 如果音量过大，需要检查是否会导致削峰
        if volume > 1.0:
            # 计算当前音频的最大绝对值
            max_val = np.abs(speech_data).max()
            if max_val > 0:
                # 确保不会超过16-bit范围
                safe_multiplier = 32767 / max_val
                adjusted_multiplier = min(adjusted_multiplier, safe_multiplier)
        
        logger.info(f"音量调整: {volume:.1f}倍, 乘数: {adjusted_multiplier:.0f}")
        
        return (speech_data * adjusted_multiplier).astype(np.int16)

    def generate_audio_stream(self, tts_text,instruct_text="请用自然流畅的语调朗读", volume=1.0, speed=1.0, buffer_size=BUFFER_SIZE,cloneprompt_speech_16k=None):
        """使用流式缓冲优化的音频生成"""
        # 重置停止标志
        self.stop_inference = False
        
        # TTS参数设置
        stream = True
        # 初始化音频缓冲区（启用流式模式）
        is_stream_mode = stream
        buffer = AudioBuffer(buffer_size=buffer_size, stream_mode=is_stream_mode)
        
        logger.info(f"初始化音频缓冲区: 流式模式={is_stream_mode}, 缓冲区大小={buffer.buffer_size}, 最小块={buffer.min_chunk_size}")
        
        logger.info(f"开始CosyVoice推理: text={tts_text[:50]}...")
        logger.info(f"使用prompt音频: 形状={self.prompt_speech_16k.shape}")
        cloneprompt_speech_16k = cloneprompt_speech_16k if cloneprompt_speech_16k is not None else self.prompt_speech_16k
        try:
            print("instruct_text:",instruct_text)
            print("tts_text:",tts_text)
            # 使用instruct2方法进行推理
            for model_output in self.cosyvoice.inference_instruct2(
                tts_text, 
                instruct_text, 
                cloneprompt_speech_16k, 
                stream=stream
            ):
                # 检查是否请求停止推理
                if self.stop_inference:
                    logger.info("推理过程被中断，停止生成音频")
                    return  # 直接返回，不再处理后续输出
                
                try:
                    # 获取音频数据并进行音量调整
                    speech_data = model_output['tts_speech'].cpu().numpy()
                    speech_data = self.apply_volume_adjustment(speech_data, volume)
                    
                    # 添加到缓冲区
                    buffer.add_data(speech_data.tobytes())
                    
                    # 检查是否需要输出
                    if buffer.should_yield():
                        # 再次检查停止标志（在输出前）
                        if self.stop_inference:
                            logger.info("推理过程被中断，停止生成音频")
                            return
                        
                        data = buffer.get_data()
                        if data:
                            # 将音频数据编码为base64
                            audio_base64 = base64.b64encode(data).decode('utf-8')
                            
                            # 构造音频数据包
                            audio_data = {
                                'data': audio_base64, 
                                'chunk_id': buffer.chunk_count,
                                'size': len(data),
                                'sample_rate': OUTPUT_SAMPLE_RATE,  # 使用统一的采样率配置
                                'channels': AUDIO_CHANNELS,
                                'data_type': 'int16',  # 音量调整后的数据类型
                                'volume': volume,
                                'speed': speed
                            }
                            yield audio_data
                            
                except Exception as e:
                    logger.error(f"处理音频数据时出错: {str(e)}")
                    continue
            
            # 输出剩余的缓冲数据（仅在未被中断时）
            if not self.stop_inference:
                remaining_data = buffer.flush()
                if remaining_data:
                    audio_base64 = base64.b64encode(remaining_data).decode('utf-8')
                    audio_data = {
                        'data': audio_base64,
                        'chunk_id': buffer.chunk_count,
                        'size': len(remaining_data),
                        'sample_rate': OUTPUT_SAMPLE_RATE,
                        'channels': AUDIO_CHANNELS,
                        'data_type': 'int16',
                        'volume': volume,
                        'speed': speed,
                        'final_chunk': True  # 标记最后一个块
                    }
                    yield audio_data
                # 输出缓冲区统计信息
                stats = buffer.get_stats()
                logger.info(f"音频生成完成，缓冲区统计: {stats}")
            else:
                logger.info("推理被中断，跳过剩余数据输出")
        except Exception as e:
            logger.error(f"CosyVoice推理过程中出错: {str(e)}")
            logger.error(traceback.format_exc())
            raise


    def generate_audio(self, text,instruct_text="请用自然流畅的语调朗读", speed=1.0,stream=False):
        voice_response = self.cosyvoice.inference_instruct2(
            text,
            instruct_text,
            self.prompt_speech_16k,
            stream=stream,
            speed=speed
        )
        # for i,j in enumerate(voice_response):
        #     torchaudio.save('instruct_{}.wav'.format(i), j['tts_speech'], self.cosyvoice.sample_rate)
        return voice_response

    def set_recorded_audio(self, audio_data):
        self.recorded_audio = audio_data
        """加载默认音频"""
        try:
            self.default_audio = load_default_audio("input.wav")
            logger.info("默认音频加载完成")
        except Exception as e:
            logger.error(f"加载默认音频失败: {str(e)}")
            self.default_audio = prepare_audio_data(None, target_length=16000)
        pass

    def register_audio_data(self):
        if self._is_registered:
            return
        for i in range(3):
            voice_response = self.generate_audio_stream("开始预热模型")
            for audio_data in voice_response:
                pass
        self._is_registered = True
        print("======================初始化成功================")

def load_default_audio(file_path="input.wav", target_length=16000):
    """加载默认音频文件"""
    try:
        if not os.path.exists(file_path):
            logger.warning(f"默认音频文件不存在: {file_path}")
            return prepare_audio_data(None, target_length)
        
        logger.info(f"加载默认音频文件: {file_path}")
        # 使用CosyVoice的load_wav函数加载音频
        _, load_wav_func = _import_cosyvoice()
        speech_16k = load_wav_func(file_path, 16000)
        
        # 确保是正确的格式
        if speech_16k.dim() == 1:
            speech_16k = speech_16k.unsqueeze(0)  # 转为 [1, samples]
        
        logger.info(f"默认音频加载成功: 形状={speech_16k.shape}")
        return speech_16k
    except Exception as e:
        logger.error(f"加载默认音频出错: {str(e)}")
        return prepare_audio_data(None, target_length)
    
def prepare_audio_data(audio_list, target_length=16000):
    """准备音频数据，确保格式正确"""
    logger.info(f"原始音频数据类型: {type(audio_list)}")
    
    if not audio_list:
        # 创建默认静音音频
        audio_list = [0.0] * target_length
        logger.info(f"使用默认静音音频，长度: {target_length}")
    
    # 转换为numpy数组
    audio_array = np.array(audio_list, dtype=np.float32)
    logger.info(f"转换为numpy数组后: 形状={audio_array.shape}, 维度={audio_array.ndim}")
    
    # 强制确保是1维数组
    if audio_array.ndim > 1:
        logger.info(f"检测到多维数组，从 {audio_array.shape} 展平为1维")
        audio_array = audio_array.flatten()
    
    # 确保输出是1维的
    if audio_array.ndim != 1:
        raise ValueError(f"音频数据必须是1维的，当前维度: {audio_array.ndim}")
    
    logger.info(f"展平后的音频数组: 形状={audio_array.shape}, 长度={len(audio_array)}")
    
    # 调整长度
    if len(audio_array) < target_length:
        # 如果太短，用零填充
        pad_length = target_length - len(audio_array)
        audio_array = np.pad(audio_array, (0, pad_length), mode='constant', constant_values=0.0)
        logger.info(f"音频太短，填充了 {pad_length} 个零")
    elif len(audio_array) > target_length:
        # 如果太长，截断到目标长度
        audio_array = audio_array[:target_length]
        logger.info(f"音频太长，截断到 {target_length}")
    
    # 转换为torch tensor，首先创建1维张量
    audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
    
    # *** 关键修复：转换为二维张量 [1, samples] 格式 ***
    # CosyVoice 的 resample 和特征提取器期望二维张量
    audio_tensor = audio_tensor.unsqueeze(0)  # 从 [samples] 变为 [1, samples]
    
    logger.info(f"最终音频张量: 形状={audio_tensor.shape}, 维度={audio_tensor.dim()}, 类型={audio_tensor.dtype}")
    logger.info(f"音频张量详细信息: channels={audio_tensor.shape[0]}, samples={audio_tensor.shape[1]}")
    
    # 最后的安全检查
    if audio_tensor.shape[0] != 1:
        logger.error(f"音频张量通道数错误！期望: 1, 实际: {audio_tensor.shape[0]}")
    if audio_tensor.shape[1] != target_length:
        logger.warning(f"音频长度不匹配！期望: {target_length}, 实际: {audio_tensor.shape[1]}")
    
    return audio_tensor


if __name__ == "__main__":
    cosyvoice = CosyVoiceTTS()
    audio_data = cosyvoice.generate_audio_stream("你好，我是小爱同学，很高兴认识你")
    print(audio_data)