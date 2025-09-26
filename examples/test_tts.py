import sys
import os
import time
import base64
import numpy as np
import traceback

# 添加项目根目录到路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from cosyVoiceTTS import CosyVoiceTTS

def save_audio_from_base64(audio_data_list, output_path):
    """将base64编码的音频数据保存为wav文件"""
    try:
        # 合并所有音频块
        all_audio_bytes = b''
        for audio_data in audio_data_list:
            audio_bytes = base64.b64decode(audio_data['data'])
            all_audio_bytes += audio_bytes
        
        # 转换为numpy数组
        audio_array = np.frombuffer(all_audio_bytes, dtype=np.int16)
        
        # 保存为wav文件 (简单的PCM格式)
        with open(output_path, 'wb') as f:
            # WAV文件头
            sample_rate = audio_data_list[0].get('sample_rate', 24000)
            channels = audio_data_list[0].get('channels', 1)
            bits_per_sample = 16
            
            # 计算文件大小
            data_size = len(all_audio_bytes)
            file_size = 36 + data_size
            
            # RIFF头
            f.write(b'RIFF')
            f.write(file_size.to_bytes(4, 'little'))
            f.write(b'WAVE')
            
            # fmt块
            f.write(b'fmt ')
            f.write((16).to_bytes(4, 'little'))  # fmt块大小
            f.write((1).to_bytes(2, 'little'))   # PCM格式
            f.write(channels.to_bytes(2, 'little'))
            f.write(sample_rate.to_bytes(4, 'little'))
            f.write((sample_rate * channels * bits_per_sample // 8).to_bytes(4, 'little'))
            f.write((channels * bits_per_sample // 8).to_bytes(2, 'little'))
            f.write(bits_per_sample.to_bytes(2, 'little'))
            
            # data块
            f.write(b'data')
            f.write(data_size.to_bytes(4, 'little'))
            f.write(all_audio_bytes)
            
        print(f"✅ 音频已保存到: {output_path}")
        return True
    except Exception as e:
        print(f"❌ 保存音频失败: {str(e)}")
        return False

def test_basic_tts(tts):
    """测试基础TTS功能"""
    print("\n" + "="*50)
    print("🎵 测试1: 基础TTS语音合成")
    print("="*50)
    
    test_text = "你好，我是CosyVoice语音合成系统，很高兴为您服务。"
    print(f"📝 测试文本: {test_text}")
    
    try:
        start_time = time.time()
        audio_chunks = []
        
        # 生成音频流
        for i, audio_data in enumerate(tts.generate_audio_stream(test_text)):
            audio_chunks.append(audio_data)
            print(f"📦 接收音频块 {i+1}: {audio_data['size']} bytes")
            
        end_time = time.time()
        print(f"⏱️ 生成时间: {end_time - start_time:.2f}秒")
        print(f"📊 总音频块数: {len(audio_chunks)}")
        
        # 保存音频
        if audio_chunks:
            save_audio_from_base64(audio_chunks, "temp_audio/test_basic.wav")
            return True
        else:
            print("❌ 未生成任何音频数据")
            return False
            
    except Exception as e:
        print(f"❌ 基础TTS测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_streaming_tts(tts):
    """测试流式TTS功能"""
    print("\n" + "="*50)
    print("🌊 测试2: 流式TTS语音合成")
    print("="*50)
    
    test_text = "这是一个流式语音合成测试。我们将逐步生成音频数据，实现低延迟的语音输出效果。"
    print(f"📝 测试文本: {test_text}")
    
    try:
        start_time = time.time()
        audio_chunks = []
        first_chunk_time = None
        
        # 生成流式音频
        for i, audio_data in enumerate(tts.generate_audio_stream(test_text, buffer_size=8192)):
            if first_chunk_time is None:
                first_chunk_time = time.time()
                print(f"⚡ 首个音频块延迟: {first_chunk_time - start_time:.3f}秒")
            
            audio_chunks.append(audio_data)
            print(f"🎵 流式音频块 {i+1}: {audio_data['size']} bytes, 块ID: {audio_data['chunk_id']}")
            
        end_time = time.time()
        print(f"⏱️ 总生成时间: {end_time - start_time:.2f}秒")
        print(f"📊 总音频块数: {len(audio_chunks)}")
        
        # 保存音频
        if audio_chunks:
            save_audio_from_base64(audio_chunks, "temp_audio/test_streaming.wav")
            return True
        else:
            print("❌ 未生成任何流式音频数据")
            return False
            
    except Exception as e:
        print(f"❌ 流式TTS测试失败: {str(e)}")
        traceback.print_exc()
        return False

def test_custom_voice_instructions(tts):
    """测试不同语音指令"""
    print("\n" + "="*50)
    print("🎭 测试3: 自定义语音指令")
    print("="*50)
    
    test_cases = [
        {
            "text": "欢迎来到智能语音助手系统！",
            "instruction": "请用热情欢快的语调朗读",
            "filename": "test_happy.wav"
        },
        {
            "text": "很抱歉，系统出现了错误，请稍后重试。",
            "instruction": "请用温和道歉的语调朗读",
            "filename": "test_apologetic.wav"
        },
        {
            "text": "今天天气晴朗，温度25度，适合户外活动。",
            "instruction": "请用播音员的专业语调朗读",
            "filename": "test_professional.wav"
        }
    ]
    
    success_count = 0
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\n📋 测试用例 {i}:")
        print(f"📝 文本: {test_case['text']}")
        print(f"🎯 指令: {test_case['instruction']}")
        
        try:
            start_time = time.time()
            audio_chunks = []
            
            # 生成音频
            for audio_data in tts.generate_audio_stream(
                test_case['text'], 
                instruct_text=test_case['instruction']
            ):
                audio_chunks.append(audio_data)
            
            end_time = time.time()
            print(f"⏱️ 生成时间: {end_time - start_time:.2f}秒")
            print(f"📦 音频块数: {len(audio_chunks)}")
            
            # 保存音频
            if audio_chunks:
                output_path = f"temp_audio/{test_case['filename']}"
                if save_audio_from_base64(audio_chunks, output_path):
                    success_count += 1
                    print(f"✅ 测试用例 {i} 成功")
                else:
                    print(f"❌ 测试用例 {i} 保存失败")
            else:
                print(f"❌ 测试用例 {i} 未生成音频")
                
        except Exception as e:
            print(f"❌ 测试用例 {i} 失败: {str(e)}")
            traceback.print_exc()
    
    print(f"\n📊 自定义指令测试结果: {success_count}/{len(test_cases)} 成功")
    return success_count == len(test_cases)

def test_parameter_variations(tts):
    """测试不同参数设置"""
    print("\n" + "="*50)
    print("⚙️ 测试4: 参数变化测试")
    print("="*50)
    
    test_text = "这是参数测试，我们将测试不同的音量和语速设置。"
    test_cases = [
        {"volume": 0.5, "speed": 0.8, "filename": "test_quiet_slow.wav"},
        {"volume": 1.0, "speed": 1.0, "filename": "test_normal.wav"},
        {"volume": 1.5, "speed": 1.2, "filename": "test_loud_fast.wav"},
    ]
    
    success_count = 0
    
    for i, params in enumerate(test_cases, 1):
        print(f"\n🔧 参数组合 {i}:")
        print(f"🔊 音量: {params['volume']}")
        print(f"⏩语速: {params['speed']}")
        
        try:
            start_time = time.time()
            audio_chunks = []
            
            # 生成音频
            for audio_data in tts.generate_audio_stream(
                test_text,
                volume=params['volume'],
                speed=params['speed']
            ):
                audio_chunks.append(audio_data)
            
            end_time = time.time()
            print(f"⏱️ 生成时间: {end_time - start_time:.2f}秒")
            print(f"📦 音频块数: {len(audio_chunks)}")
            
            # 保存音频
            if audio_chunks:
                output_path = f"temp_audio/{params['filename']}"
                if save_audio_from_base64(audio_chunks, output_path):
                    success_count += 1
                    print(f"✅ 参数组合 {i} 成功")
                else:
                    print(f"❌ 参数组合 {i} 保存失败")
            else:
                print(f"❌ 参数组合 {i} 未生成音频")
                
        except Exception as e:
            print(f"❌ 参数组合 {i} 失败: {str(e)}")
            traceback.print_exc()
    
    print(f"\n📊 参数测试结果: {success_count}/{len(test_cases)} 成功")
    return success_count == len(test_cases)

def create_temp_directory():
    """创建临时音频目录"""
    temp_dir = "temp_audio"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
        print(f"📁 创建临时目录: {temp_dir}")
    return temp_dir

if __name__ == '__main__':
    print("🚀 开始CosyVoice TTS测试")
    print("="*60)
    
    # 创建临时目录
    create_temp_directory()
    
    try:
        # 初始化TTS引擎
        print("🔧 初始化TTS引擎...")
        start_init = time.time()
        model_path = "iic/CosyVoice2-0.5B"
        tts = CosyVoiceTTS(model_path)
        end_init = time.time()
        print(f"✅ TTS引擎初始化完成，耗时: {end_init - start_init:.2f}秒")
        
        # 预热模型
        print("\n🔥 预热模型...")
        tts.register_audio_data()
        
        # 运行测试
        test_results = []
        
        # 测试1: 基础TTS
        test_results.append(("基础TTS", test_basic_tts(tts)))
        
        # 测试2: 流式TTS
        test_results.append(("流式TTS", test_streaming_tts(tts)))
        
        # 测试3: 自定义指令
        test_results.append(("自定义指令", test_custom_voice_instructions(tts)))
        
        # 测试4: 参数变化
        test_results.append(("参数变化", test_parameter_variations(tts)))
        
        # 显示测试结果
        print("\n" + "="*60)
        print("📋 测试结果总结")
        print("="*60)
        
        passed = 0
        total = len(test_results)
        
        for test_name, result in test_results:
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name:<15} : {status}")
            if result:
                passed += 1
        
        print(f"\n📊 总体结果: {passed}/{total} 测试通过")
        
        if passed == total:
            print("🎉 所有测试都通过了！TTS系统工作正常。")
        else:
            print("⚠️ 部分测试失败，请检查系统配置。")
            
        print(f"\n📁 生成的音频文件保存在: temp_audio/ 目录")
        
    except Exception as e:
        print(f"❌ TTS测试过程中出现严重错误: {str(e)}")
        traceback.print_exc()