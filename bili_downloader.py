"""
A tool for downloading videos and danmaku from Bilibili
Copyright (C) 2025 dowdah

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""
import json
import os
import time
import xmltodict
import asyncio
from bilibili_api import Geetest, GeetestType, login_v2, sync, Credential, bangumi, HEADERS, get_client, video


BASE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
FFMPEG_PATH = 'ffmpeg'
cookies_path = os.path.join(BASE_DIR, "cookies.json")


async def login() -> Credential:
    choice = input("pwd(p) / sms(s) / qr code(q):")
    if not choice in ["p", "s", "q"]:
        raise ValueError("Invalid choice")

    if choice in ["p", "s"]:
        gee = Geetest()                                                         # 实例化极验测试类
        await gee.generate_test()                                               # 生成测试
        gee.start_geetest_server()                                              # 在本地部署网页端测试服务
        print(gee.get_geetest_server_url())                                     # 获取本地服务链接
        while not gee.has_done():                                               # 如果测试未完成
            pass                                                                # 就等待
        gee.close_geetest_server()                                              # 关闭部署的网页端测试服务
        print("result:", gee.get_result())

        # 1. 密码登录
        if choice == "p":
            username = input("username:")                                       # 手机号/邮箱
            password = input("password:")                                       # 密码
            cred = await login_v2.login_with_password(
                username=username, password=password, geetest=gee               # 调用接口登陆
            )

        # 2. 验证码登录
        if choice == "s":
            phone = login_v2.PhoneNumber(input("phone:"), "+86")                # 实例化手机号类
            captcha_id = await login_v2.send_sms(phonenumber=phone, geetest=gee)# 发送验证码
            print("captcha_id:", captcha_id)                                    # 顺便获得对应的 captcha_id
            code = input("code: ")
            cred = await login_v2.login_with_sms(
                phonenumber=phone, code=code, captcha_id=captcha_id             # 调用接口登陆
            )

    # 3. 扫码登录
    if choice == "q":
        qr = login_v2.QrCodeLogin(platform=login_v2.QrCodeLoginChannel.WEB)  # 生成二维码登录实例，平台选择网页端
        await qr.generate_qrcode()  # 生成二维码
        print(qr.get_qrcode_terminal())  # 生成终端二维码文本，打印
        while not qr.has_done():  # 在完成扫描前轮询
            print(await qr.check_state())  # 检查状态
            time.sleep(1)  # 轮训间隔建议 >=1s
        cred = qr.get_credential()

    # 安全验证
    if isinstance(cred, login_v2.LoginCheck):
        # 如法炮制 Geetest
        gee = Geetest()                                                     # 实例化极验测试类
        await gee.generate_test(type_=GeetestType.VERIFY)                   # 生成测试 (注意 type_ 为 GeetestType.VERIFY)
        gee.start_geetest_server()                                          # 在本地部署网页端测试服务
        print(gee.get_geetest_server_url())                                 # 获取本地服务链接
        while not gee.has_done():                                           # 如果测试未完成
            pass                                                            # 就等待
        gee.close_geetest_server()                                          # 关闭部署的网页端测试服务
        print("result:", gee.get_result())
        await cred.send_sms(gee)                                            # 发送验证码
        code = input("code:")
        cred = await cred.complete_check(code)                              # 调用接口登陆

    return cred


def xml_2_json(xml_str:str):
    """
    将B站的XML弹幕数据转换为JSON格式，供NPlayer使用。
    :param xml_str: B站的XML弹幕数据字符串
    :return: tuple (json_out, len_danmu)
    """
    parsed_data = xmltodict.parse(xml_str)
    json_conversion = parsed_data['i']['d']
    json_out = {'code': 1, 'data': []}
    for danmu in json_conversion:
        list_param = danmu['@p'].split(',')
        hex_color = hex(int(list_param[3]))[2:]
        hex_color = '#' + '0' * (6 - len(hex_color)) + hex_color
        one_danmaku = {"author": list_param[6], "time": float(list_param[0]), "text": danmu['#text'],
                       "color": hex_color, "type": "scroll"}
        if list_param[1] == '4':
            one_danmaku['type'] = "bottom"
        if list_param[1] == '5':
            one_danmaku['type'] = "top"
        json_out['data'].append(one_danmaku)
    json_out['data'] = sorted(json_out['data'], key=lambda x: x['time'])
    return json_out, len(json_conversion)


async def download(url: str, out: str, intro: str):
    dwn_id = await get_client().download_create(url, HEADERS)
    bts = 0
    tot = get_client().download_content_length(dwn_id)
    with open(out, "wb") as file:
        while True:
            print(f"{intro} - {out} [{bts} / {tot}]", end="\r")
            bts += file.write(await get_client().download_chunk(dwn_id))
            if bts == tot:
                break
    print()


async def download_bangumi(media_id, cred):
    output_dir = os.path.join(BASE_DIR, "bangumi", str(media_id))
    # 实例化 Bangumi 类
    b = bangumi.Bangumi(media_id=media_id, credential=cred)
    # 获取所有剧集
    for idx, ep in enumerate(await b.get_episodes()):
        await download_episode(ep, os.path.join(output_dir, f"{idx}.mp4"))
        await download_ep_danmaku(ep, os.path.join(output_dir, f"dms_raw_{idx}.json"))


async def download_episode(ep: bangumi.Episode, out: str):
    print(f"########## {await ep.get_bvid()} ##########")
    # 获取视频下载链接
    download_url_data = await ep.get_download_url()
    # 解析视频下载信息
    detecter = video.VideoDownloadURLDataDetecter(data=download_url_data)
    streams = detecter.detect_best_streams(
        video_max_quality=video.VideoQuality._8K,
        audio_max_quality=video.AudioQuality._192K,
        no_dolby_audio=False,
        no_dolby_video=False,
        no_hdr=False,
        no_hires=False
    )
    # 有 MP4 流 / FLV 流两种可能
    if detecter.check_video_and_audio_stream():
        # MP4 流下载
        await download(streams[0].url, "video_temp.m4s", "视频流")
        await download(streams[1].url, "audio_temp.m4s", "音频流")
        # 混流
        os.system(
            f"{FFMPEG_PATH} -i video_temp.m4s -i audio_temp.m4s -vcodec copy -acodec copy {out}"
        )
        # 删除临时文件
        os.remove("video_temp.m4s")
        os.remove("audio_temp.m4s")
    else:
        # FLV 流下载
        await download(streams[0].url, "flv_temp.flv", "FLV 音视频流")
        # 转换文件格式
        os.system(f"{FFMPEG_PATH} -i flv_temp.flv {out}")
        # 删除临时文件
        os.remove("flv_temp.flv")

    print(f"已下载为：{out}")


async def download_ep_danmaku(ep: bangumi.Episode, out: str):
    """
    下载番剧的弹幕
    :param ep: 番剧剧集对象
    :param out: 输出文件路径
    """
    dms = await ep.get_danmaku_xml()
    dms_json, len_danmu = xml_2_json(dms)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(dms_json, f, ensure_ascii=False, indent=4)
    print(f"已下载弹幕：{out}，共 {len_danmu} 条")


async def get_bangumi_info(bangumi_id: int, cred: Credential):
    """
    获取番剧信息
    :param bangumi_id: 番剧ID
    :param cred: 登录凭证
    :return: 番剧信息字典
    """
    b = bangumi.Bangumi(bangumi_id, credential=cred)
    eps = await b.get_episodes()
    ep_info = {ep.get_epid(): await ep.get_episode_info() for ep in eps}
    return ep_info


if __name__ == "__main__":
    if os.path.exists(cookies_path):
        with open(cookies_path, "r", encoding="utf-8") as f:
            cookies_dict = json.load(f)
        cred = Credential(sessdata=cookies_dict["SESSDATA"], bili_jct=cookies_dict["bili_jct"],
                          ac_time_value=cookies_dict["ac_time_value"])
    else:
        cookies_dict = {}
    if cookies_dict and sync(cred.check_valid()):
        if sync(cred.check_refresh()):
            sync(cred.refresh())
    else:
        cred = sync(login())
        with open(cookies_path, "w", encoding="utf-8") as f:
            json.dump(cred.get_cookies(), f, ensure_ascii=False, indent=4)
    asyncio.run(download_bangumi(28235123, cred))  # 替换为实际的番剧ID
