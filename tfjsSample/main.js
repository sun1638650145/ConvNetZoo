;import * as tools from "./backend";

function main() {
    tools.showVersion();
    // 加载模型
    tools.loadModel();

    // 添加元素
    let filesElement = document.createElement('input');
    filesElement.setAttribute('type', 'file');
    document.body.appendChild(filesElement);

    // 创建监听函数
    filesElement.addEventListener('change', tools.selectImage, false);
}

main();