#把xml文件中的中文转换为英文名
import os
import xml.etree.ElementTree as ET


def replace_names_in_xml(folder_path, replacements):
    # 遍历文件夹中的所有文件
    for filename in os.listdir(folder_path):
        if filename.endswith('.xml'):
            file_path = os.path.join(folder_path, filename)

            # 解析 XML 文件
            tree = ET.parse(file_path)
            root = tree.getroot()

            # 遍历 XML 树，替换指定名称
            for elem in root.iter():
                if elem.text in replacements:
                    print(f"Replacing '{elem.text}' with '{replacements[elem.text]}' in {filename}")
                    elem.text = replacements[elem.text]

            # 将修改后的 XML 保存回文件
            tree.write(file_path, encoding="utf-8", xml_declaration=False)


# 使用示例
folder_path = 'anno'  # 替换为你的文件夹路径
replacements = {
    # '破洞': '0',
    # '水渍': '1',
    # '油渍': '1',
    # '污渍': '1',
    # '三丝': '2',
    # '结头': '3',
    # '花板跳': '4',
    # '百脚': '5',
    # '毛粒': '6',
    # '粗经': '7',
    # '松经': '8',
    # '断经': '9',
    # '吊经': '10',
    # '粗维': '11',
    # '纬缩': '12',
    # '浆斑': '13',
    # '整经结': '14',
    # '星跳': '15',
    # '跳花': '16',####这里应该是15，后面往前顺延一个，已经改掉
    # '断氨纶': '17',
    # '稀密档': '18',
    # '浪纹档': '18',
    # '色差档': '18',
    # '磨痕': '19',
    # '轧痕': '19',
    # '修痕': '19',
    # '烧毛痕': '19',
    # '死皱': '20',
    # '云织': '20',
    # '双纬': '20',
    # '双经': '20',
    # '跳纱': '20',
    # '筘路': '20',
    # '纬纱不良': '20',
    '\u7834\u6d1e': '0',
    '\u6c34\u6e0d': '1',
    '\u6cb9\u6e0d': '1',
    '\u6c61\u6e0d': '1',
    '\u4e09\u4e1d': '2',
    '\u7ed3\u5934': '3',
    '\u82b1\u677f\u8df3': '4',
    '\u767e\u811a': '5',
    '\u6bdb\u7c92': '6',
    '\u7c97\u7ecf': '7',
    '\u677e\u7ecf': '8',
    '\u65ad\u7ecf': '9',
    '\u540a\u7ecf': '10',
    '\u7c97\u7ef4': '11',
    '\u7eac\u7f29': '12',
    '\u6d46\u6591': '13',
    '\u6574\u7ecf\u7ed3': '14',
    '\u661f\u8df3': '15',
    '\u8df3\u82b1': '16',
    '\u65ad\u6c28\u7eb6': '17',
    '\u7a00\u5bc6\u6863': '18',
    '\u6d6a\u7eb9\u6863': '18',
    '\u8272\u5dee\u6863': '18',
    '\u78e8\u75d5': '19',
    '\u8f67\u75d5': '19',
    '\u4fee\u75d5': '19',
    '\u70e7\u6bdb\u75d5': '19',
    '\u6b7b\u76b1': '20',
    '\u4e91\u7ec7': '20',
    '\u53cc\u7eac': '20',
    '\u53cc\u7ecf': '20',
    '\u8df3\u7eb1': '20',
    '\u7b58\u8def': '20',
    '\u7eac\u7eb1\u4e0d\u826f': '20',

}

replace_names_in_xml(folder_path, replacements)
