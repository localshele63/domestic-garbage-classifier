CLASS_NAMES = ["dry", "hazardous", "recyclable", "wet"]

CLASS_LABELS = {
    "recyclable": "可回收物",
    "hazardous": "有害垃圾",
    "wet": "湿垃圾/厨余垃圾",
    "dry": "干垃圾/其他垃圾",
}

CLASS_EXAMPLES = {
    "recyclable": "纸箱、报纸、塑料瓶、玻璃瓶、金属罐",
    "hazardous": "电池图片（训练数据未区分电池化学类型）",
    "wet": "biological类图片（如食物残余、果皮、菜叶）",
    "dry": "衣物、鞋类及trash类图片（沿用工程标签）",
}


def display_name(name):
    label = CLASS_LABELS.get(name, name)
    return f"{label} ({name})"
