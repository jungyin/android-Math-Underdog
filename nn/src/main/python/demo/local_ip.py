import requests

ali_map_key = "c49485098e88a45c418addbbfe4de800"




def get_location():
    try:
        # 使用 ip-api 获取当前位置信息
        response = requests.get("http://ip-api.com/json/?fields=status,message,country,regionName,city,district,zip,lat,lon,query", timeout=10)
        response.raise_for_status()  # 检查请求是否成功
        data = response.json()

        if data['status'] == 'fail':
            print("无法获取位置信息:", data['message'])
            return None
        # 打印获取的信息
        print(f"IP地址: {data['query']}")
        print(f"国家: {data['country']}")
        print(f"地区: {data['regionName']}")
        print(f"城市: {data['city']}")
        print(f"区/县: {data['district']}")
        print(f"邮政编码: {data['zip']}")
        print(f"纬度: {data['lat']}")
        print(f"经度: {data['lon']}")

        return data

    except requests.RequestException as e:
        print(f"请求错误: {e}")
        return None


def get_location_amap(key, sig=None):
    """
    使用高德地图IP定位API获取当前IP地理位置信息（精确到县级）

    :param key: 高德开放平台申请的Web服务Key
    :param sig: 签名参数（如果启用了安全密钥）
    :return: 解析后的JSON数据，失败返回None
    """
    url = "https://restapi.amap.com/v3/ip"

    # 构造请求参数
    params = {
        "key": key
    }
    if sig:
        params["sig"] = sig

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # 检查HTTP错误
        data = response.json()

        if data.get("status") == "1":
            print("请求成功！")
            return data
        else:
            print("接口返回失败:", data.get("info", "未知错误"))
            return None

    except requests.RequestException as e:
        print("网络请求异常:", e)
        return None
def get_weather_amap(key,citycode,extensions='base', sig=None):
    """
    使用高德地图IP天气API获取指定位置id的天气情况

    :param key: 高德开放平台申请的Web服务Key
    :param citycode: 位置id
    :param extensions: 天气类型,base为实时,all 为当天,默认base
    :return: 解析后的JSON数据,失败返回None
    """
    url = "https://restapi.amap.com/v3/weather/weatherInfo"

    # 构造请求参数
    params = {
        "key": key,
        "city": citycode,
        "extensions": extensions,
        "output": "JSON"
    }
    if sig:
        params["sig"] = sig

    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()  # 检查HTTP错误
        data = response.json()

        if data.get("status") == "1":
            print("请求成功！")
            return data
        else:
            print("接口返回失败:", data.get("info", "未知错误"))
            return None

    except requests.RequestException as e:
        print("网络请求异常:", e)
        return None





if __name__ == "__main__":
    location_info = get_location_amap(ali_map_key)
    print(location_info)
    location_info = get_weather_amap(ali_map_key,location_info["adcode"])
    print(location_info)