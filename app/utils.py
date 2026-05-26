"""工具函数"""

import re
from typing import Optional
from .config import MimoAccount


def parse_curl(curl_command: str) -> Optional[MimoAccount]:
    """
    解析cURL命令提取Mimo账号凭证

    Args:
        curl_command: cURL命令字符串

    Returns:
        MimoAccount对象或None
    """
    account = {
        'service_token': '',
        'user_id': '',
        'xiaomichatbot_ph': ''
    }

    # 提取cookies（支持多种格式）
    cookie_match = re.search(r"(?:-b|--cookie)\s+'([^']+)'", curl_command)
    if not cookie_match:
        cookie_match = re.search(r'(?:-b|--cookie)\s+"([^"]+)"', curl_command)
    if not cookie_match:
        cookie_match = re.search(r"-H\s+'[Cc]ookie:\s*([^']+)'", curl_command)
    if not cookie_match:
        cookie_match = re.search(r'-H\s+"[Cc]ookie:\s*([^"]+)"', curl_command)
    if not cookie_match:
        return None

    cookies = cookie_match.group(1)

    # 提取serviceToken
    service_token_match = re.search(r'serviceToken="([^"]+)"', cookies)
    if service_token_match:
        account['service_token'] = service_token_match.group(1)

    # 提取userId
    user_id_match = re.search(r'userId=(\d+)', cookies)
    if user_id_match:
        account['user_id'] = user_id_match.group(1)

    # 提取xiaomichatbot_ph
    ph_match = re.search(r'xiaomichatbot_ph="([^"]+)"', cookies)
    if ph_match:
        account['xiaomichatbot_ph'] = ph_match.group(1)

    # 验证必需字段
    if not account['service_token']:
        return None

    return MimoAccount(**account)


def safe_utf8_len(text: str, max_len: int) -> int:
    """
    安全的UTF-8字符串长度计算，避免在多字节字符中间截断

    Args:
        text: 文本字符串
        max_len: 最大长度

    Returns:
        安全的截断长度
    """
    if max_len <= 0 or max_len >= len(text):
        return len(text)

    # Python 3的字符串是Unicode，不需要特殊处理UTF-8边界
    # 但为了与Go版本保持一致的逻辑，我们保留这个函数
    return max_len


def build_query_from_messages(messages: list) -> tuple:
    """
    从消息列表构建查询字符串，并解析配置标签

    Returns:
        (query_string, thinking, search)
    """
    thinking = False
    search = False

    query_parts = []
    for msg in messages:
        content = msg.content

        # 解析标签: [think=on/off], [search=on/off]
        if "[think=on]" in content:
            thinking = True
        if "[think=off]" in content:
            thinking = False
        if "[search=on]" in content:
            search = True
        if "[search=off]" in content:
            search = False

        # 移除标签以保持给模型的输入干净
        clean_content = content
        clean_content = clean_content.replace("[think=on]", "").replace("[think=off]", "")
        clean_content = clean_content.replace("[search=on]", "").replace("[search=off]", "")
        clean_content = clean_content.strip()

        if clean_content:
            query_parts.append(f"{msg.role}: {clean_content}")

    return "\n".join(query_parts), thinking, search
