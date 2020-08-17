;import "regenerator-runtime/runtime";
import * as tf from "@tensorflow/tfjs";

let model;
let MODEL_PATH = 'https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json';

// 查看tfjs版本
function showVersion() {
    console.log("INFO:version:" + tf.version.tfjs);
}
// 图片预处理
function preprocessImage(imageElement) {
    return tf.tidy(() => {
        let offset = tf.scalar(127.5);

        let img = tf.browser.fromPixels(imageElement).toFloat();
        // WARNING:归一化必须做在resize之前
        let normalized = img.sub(offset).div(offset);
        let resized = tf.image.resizeBilinear(normalized, [224, 224]);

        console.log("INFO:图片预处理成功");
        return resized.expandDims(0);
    });
}
// 加载模型
async function loadModel() {
    model = await tf.loadLayersModel(MODEL_PATH);
    // model.summary();
}
// 模型推理
async function modelPredict(image) {
    let max = -1;
    let index = -1;

    let tensor = preprocessImage(image);
    let ans = model.predict(tensor);
    let values = await ans.data();

    // 找出最大值
    const valuesAndIndices = [];
    for (let i = 0; i < values.length; i ++) {
        valuesAndIndices.push({value: values[i], index: i});
    }
    for (let i = 0; i < valuesAndIndices.length; i ++) {
        if (valuesAndIndices[i].value > max) {
            max = valuesAndIndices[i].value;
            index = valuesAndIndices[i].index;
        }
    }
    alert("max:"+max+", index:"+index);
}
// 选择图片
function selectImage(event) {
    let files = event.target.files;
    for (let i = 0, file; file = files[i]; i ++) {
        if (!file.type.match('image.*')) {
            alert("ERROR:请选择一张图片");
            continue;
        }
        let reader = new FileReader();
        reader.onload = async function (event) {
            // 添加元素
            let imageElement = document.createElement('img');
            imageElement.setAttribute('id', 'image');
            imageElement.setAttribute('width', '224');
            imageElement.setAttribute('height', '224');
            imageElement.src = event.target.result;
            document.body.appendChild(imageElement);

            console.log("INFO:图片加载成功");
            modelPredict(imageElement);
        };
        reader.readAsDataURL(file);
    }
}

export {showVersion, preprocessImage, loadModel, modelPredict, selectImage};