import base64
import traceback
import numpy as np
import torch
import torchaudio
import logging
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice/third_party/Matcha-TTS'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'CosyVoice'))

# 延迟导入CosyVoice模块
load_wav = None

def _import_cosyvoice():
    """延迟导入cosyvoice模块"""
    global load_wav
    if load_wav is None:
        try:
            from cosyvoice.utils.file_utils import load_wav as _load_wav
            load_wav = _load_wav
        except ImportError as e:
            print(f"❌ CloneForTTS 导入 cosyvoice.utils.file_utils 失败: {e}")
            raise
    return load_wav


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


class CloneForTTS:
    def __init__(self):
        pass
        
    def decode_audio_data(self, user_id, audio_data_b64):
        """解码前端发送的音频数据并永久保存到本地"""
        try:
            # 解码base64数据
            audio_bytes = base64.b64decode(audio_data_b64)
            logger.info(f"接收到音频数据大小: {len(audio_bytes)} bytes")
            
            # 分析音频格式
            format_type, format_desc = self.analyze_audio_format(audio_bytes)
            logger.info(f"音频格式分析: {format_type} - {format_desc}")
            
            # 创建永久存储目录
            import os
            import time
            voices_dir = os.path.join(os.path.dirname(__file__), '..', 'voices')
            if not os.path.exists(voices_dir):
                os.makedirs(voices_dir)
            
            # 方法1: 尝试使用torchaudio直接解码
            try:
                logger.info("方法1: 尝试使用torchaudio解码音频数据")
                # 根据格式分析选择合适的扩展名
                if format_type == "wav":
                    suffixes = ['.wav']
                elif format_type == "webm":
                    suffixes = ['.webm', '.wav']
                elif format_type == "mp4":
                    suffixes = ['.mp4', '.wav']
                elif format_type == "ogg":
                    suffixes = ['.ogg', '.wav']
                else:
                    suffixes = ['.wav', '.webm', '.mp4', '.ogg']
                
                # 创建永久文件，尝试多种格式
                for suffix in suffixes:
                    # 使用项目路径创建永久文件（无时间戳）
                    file_name = f"user_{user_id}{suffix}"
                    file_path = os.path.join(voices_dir, file_name)
                    
                    # 写入音频数据（会覆盖同名文件）
                    with open(file_path, 'wb') as audio_file:
                        audio_file.write(audio_bytes)
                    
                    try:
                        # 使用torchaudio加载音频
                        waveform, sample_rate = torchaudio.load(file_path)
                        logger.info(f"torchaudio加载成功: 格式={suffix}, 采样率={sample_rate}, 形状={waveform.shape}")
                        
                        # 转换采样率到16kHz
                        if sample_rate != 16000:
                            resampler = torchaudio.transforms.Resample(sample_rate, 16000)
                            waveform = resampler(waveform)
                            logger.info(f"重采样到16kHz: 新形状={waveform.shape}")
                        
                        # 如果是多声道，取第一个声道
                        if waveform.shape[0] > 1:
                            waveform = waveform[0:1, :]
                            logger.info(f"转换为单声道: 形状={waveform.shape}")
                        
                        # 确保音频长度合理（限制在30秒内）
                        max_length = 30 * 16000  # 30秒
                        if waveform.shape[1] > max_length:
                            waveform = waveform[:, :max_length]
                            logger.warning(f"音频过长，已截断到30秒")
                        
                        # 确保音频长度至少有1秒
                        min_length = 16000  # 1秒
                        if waveform.shape[1] < min_length:
                            # 重复音频直到达到最小长度
                            repeat_times = (min_length + waveform.shape[1] - 1) // waveform.shape[1]
                            waveform = waveform.repeat(1, repeat_times)[:, :min_length]
                            logger.info(f"音频过短，已重复到1秒")
                        
                        logger.info(f"音频解码成功: 形状={waveform.shape}, 时长={waveform.shape[1]/16000:.2f}秒, 保存路径={file_path}")
                        
                        return waveform, file_path
                        
                    except Exception as e:
                        logger.debug(f"格式{suffix}解码失败: {str(e)}")
                        
                        # 尝试FFmpeg解码
                        if suffix in ['.webm', '.mp4']:
                            logger.info(f"尝试使用FFmpeg解码 {suffix}")
                            ffmpeg_result = self.try_ffmpeg_decode(audio_bytes, file_path)
                            if ffmpeg_result is not None:
                                logger.info("FFmpeg解码成功")
                                return ffmpeg_result, file_path
                        
                        # 如果失败，删除无效文件
                        try:
                            os.unlink(file_path)
                        except:
                            pass
                        continue
                
                logger.warning("方法1: torchaudio所有格式都解码失败")
                
            except Exception as e:
                logger.warning(f"方法1失败: {str(e)}")
            
            # 方法2: 尝试使用CosyVoice的load_wav（仅支持wav格式）
            if format_type == "wav":
                try:
                    logger.info("方法2: 尝试使用CosyVoice load_wav解码")
                    
                    # 使用项目路径创建永久WAV文件（无时间戳）
                    file_name = f"user_{user_id}.wav"
                    file_path = os.path.join(voices_dir, file_name)
                    
                    with open(file_path, 'wb') as audio_file:
                        audio_file.write(audio_bytes)
                    
                    load_wav_func = _import_cosyvoice()
                    audio_tensor = load_wav_func(file_path, 16000)
                    
                    # 确保是二维张量 [1, samples]
                    if audio_tensor.dim() == 1:
                        audio_tensor = audio_tensor.unsqueeze(0)
                    elif audio_tensor.dim() > 2:
                        audio_tensor = audio_tensor[0:1, :]
                    
                    logger.info(f"方法2: CosyVoice解码成功: 形状={audio_tensor.shape}, 保存路径={file_path}")
                    return audio_tensor, file_path
                    
                except Exception as e:
                    logger.warning(f"方法2失败: {str(e)}")
                    # 删除无效文件
                    try:
                        os.unlink(file_path)
                    except:
                        pass
            
            # 方法4: 备用PCM解码方法 (similar updates: save permanently if successful)
            try:
                logger.info("方法4: 尝试使用备用PCM解码方法")
                
                # 尝试不同的音频格式
                for dtype in [np.float32, np.int16, np.int32]:
                    try:
                        audio_array = np.frombuffer(audio_bytes, dtype=dtype).astype(np.float32)
                        
                        # 归一化处理
                        if dtype == np.int16:
                            audio_array = audio_array / 32767.0
                        elif dtype == np.int32:
                            audio_array = audio_array / 2147483647.0
                        
                        # 确保在合理范围内
                        audio_array = np.clip(audio_array, -1.0, 1.0)
                        
                        # 检查数据有效性
                        if len(audio_array) < 1000:  # 至少需要一些数据
                            continue
                        
                        # 转换为torch tensor
                        audio_tensor = torch.tensor(audio_array, dtype=torch.float32)
                        if audio_tensor.dim() == 1:
                            audio_tensor = audio_tensor.unsqueeze(0)
                        
                        # 假设是16kHz采样率，调整长度
                        min_length = 16000  # 1秒
                        if audio_tensor.shape[1] < min_length:
                            repeat_times = (min_length + audio_tensor.shape[1] - 1) // audio_tensor.shape[1]
                            audio_tensor = audio_tensor.repeat(1, repeat_times)[:, :min_length]
                        
                        logger.info(f"方法4: 备用PCM解码成功，使用dtype={dtype}: 形状={audio_tensor.shape}")
                        
                        # 创建永久文件（无时间戳）
                        file_name = f"user_{user_id}.pcm"
                        file_path = os.path.join(voices_dir, file_name)
                        
                        with open(file_path, 'wb') as audio_file:
                            audio_file.write(audio_array.tobytes()) # 写入原始PCM数据
                        
                        return audio_tensor, file_path
                        
                    except Exception:
                        continue
                
                logger.error("所有解码方法都失败了")
                return None, None
                
            except Exception as e2:
                logger.error(f"方法4失败: {str(e2)}")
                return None, None
        except Exception as e:
            logger.error(f"解码音频数据出错: {str(e)}")
            logger.error(traceback.format_exc())
            return None, None

    def try_ffmpeg_decode(self,audio_bytes, temp_file_path):
        """尝试使用FFmpeg解码音频"""
        try:
            import subprocess
            
            # 检查ffmpeg是否可用（包括本地项目目录中的ffmpeg）
            ffmpeg_paths = ['ffmpeg', './ffmpeg.exe', './ffmpeg']
            ffmpeg_cmd = None
            
            for path in ffmpeg_paths:
                try:
                    subprocess.run([path, '-version'], capture_output=True, check=True)
                    ffmpeg_cmd = path
                    logger.info(f"找到FFmpeg: {path}")
                    break
                except (subprocess.CalledProcessError, FileNotFoundError):
                    continue
            
            if ffmpeg_cmd is None:
                logger.debug("FFmpeg不可用")
                return None
            
            # 使用ffmpeg转换为WAV
            output_path = temp_file_path.replace('.webm', '_converted.wav').replace('.mp4', '_converted.wav')
            
            cmd = [
                ffmpeg_cmd, '-y',  # -y 覆盖输出文件
                '-i', temp_file_path,  # 输入文件
                '-ar', '16000',  # 采样率16kHz
                '-ac', '1',      # 单声道
                '-acodec', 'pcm_s16le',  # PCM编码
                output_path
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0 and os.path.exists(output_path):
                logger.info(f"FFmpeg转换成功: {output_path}")
                
                # 使用torchaudio加载转换后的文件
                waveform, sample_rate = torchaudio.load(output_path)
                
                # 清理转换后的文件
                try:
                    os.unlink(output_path)
                except:
                    pass
                
                return waveform
            else:
                logger.debug(f"FFmpeg转换失败: {result.stderr}")
                return None
                
        except Exception as e:
            logger.debug(f"FFmpeg解码出错: {str(e)}")
            return None

    def analyze_audio_format(self,audio_bytes):
        """分析音频数据格式"""
        try:
            # 检查文件头来判断格式
            if len(audio_bytes) < 12:
                return "unknown", "数据太短"
            
            # WAV文件头
            if audio_bytes[:4] == b'RIFF' and audio_bytes[8:12] == b'WAVE':
                return "wav", "WAV格式"
            
            # WebM文件头 
            if audio_bytes[:4] == b'\x1a\x45\xdf\xa3':
                return "webm", "WebM格式"
            
            # MP4文件头
            if b'ftyp' in audio_bytes[:32]:
                return "mp4", "MP4格式"
            
            # OGG文件头
            if audio_bytes[:4] == b'OggS':
                return "ogg", "OGG格式"
            
            # 检查是否可能是原始PCM数据
            if len(audio_bytes) > 1000:
                return "pcm", "可能是PCM原始数据"
            
            return "unknown", f"未知格式，头部: {audio_bytes[:16].hex()}"
            
        except Exception as e:
            return "error", f"分析出错: {str(e)}"