
def confirm_tag(tag):
    """实现一个函数，函数接收一个默认值，让用户输入此次任务的tag，如果直接回车，则返回默认值"""
    if tag:
        new_tag = input(f"Please confirm tag (default: {tag}) >>> ")
        if new_tag:
            tag = new_tag
    else:
        tag = input("Please input tag >>> ")

    assert tag, "Tag can not be empty!"

    return tag

if __name__ == "__main__":
    tag = "ner"
    tag = confirm_tag(tag)
    print(tag)