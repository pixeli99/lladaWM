import os
import yaml

# 指定目录路径
directory_path = "/lpai/volumes/ad-vla-vol-ga/lipengxiang/liauto_test_data_cache"

# 遍历目录中的文件名
file_names = os.listdir(directory_path)

class NoAliasDumper(yaml.SafeDumper):
    def ignore_aliases(self, data):
        return True

# 添加字符串表示器，去除引号
def str_presenter(dumper, data):
    # 不管是数字字符串还是普通字符串，都不使用引号
    return dumper.represent_scalar('tag:yaml.org,2002:str', data, style='')

# 注册自定义表示器
NoAliasDumper.add_representer(str, str_presenter)

# 提取文件名并生成 YAML 格式数据
yaml_data = {
    "liauto_train_logs": file_names  # 直接使用文件名列表
}

# 输出 YAML 文件
output_file = "liauto_train_val_test_log_split.yaml"
with open(output_file, "w") as file:
    yaml.dump(yaml_data, file, default_flow_style=False, allow_unicode=True, Dumper=NoAliasDumper)

print(f"YAML 文件已生成: {output_file}")
