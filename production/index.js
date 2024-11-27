let normalizationData = null;
let allow = 1
let MedianHouseValue = document.querySelector(".median_house_value_container")
let loadingImage = document.querySelector(".loading-image-container")

async function predict(inputFeatures,path, key) {

    const session = await ort.InferenceSession.create(path);

    const input = new Float32Array(inputFeatures);
    const tensor = new ort.Tensor('float32', input, [1, inputFeatures.length]);

    const feeds = {};
    feeds[key] = tensor

    const result = await session.run(feeds);

    return result;
}

async function loadNormalizationInfo() {
  if (normalizationData) {
    return normalizationData;
  }

  const response = await fetch('./models/normalization_info.json');
  normalizationData = await response.json(); 
  return normalizationData;
}

async function normalizeInputs(inputs) {
  const normalizationJsonInfo = await loadNormalizationInfo(); 

  let result = { status: true, message: '', data: [] };

  for (let i = 0; i < inputs.length; i++) {
    const input = inputs[i];

    if (input.value == null || input.value == undefined || input.value === '') {
      result.status = false;
      result.message = 'The ' + input.getAttribute('placeholder') + ' is empty.';
      break;
    } 

    let input_min_value = normalizationJsonInfo.min_values[input.getAttribute("data-key")]
    let input_max_value = normalizationJsonInfo.max_values[input.getAttribute("data-key")]

    let normalizeInput = (input.value - input_min_value) / (input_max_value - input_min_value);
    result.data.push(normalizeInput)
  }

  return result;
}

function formatValue(value) {

  if (value.includes('.')) {
    return false;
  }

  const valueStr = String(value).replace(/,/g, ''); 

  if (valueStr.length > 3) {
    return `${valueStr.slice(0, -3)}.${valueStr.slice(-3)}`;
  } else {
    return `0.${valueStr.padStart(3, '0')}`;
  }
}

async function deNormalizeValue(normalizedValue, target) {

    const normalizationJsonInfo = await loadNormalizationInfo();
  
    let value_min_value = normalizationJsonInfo.min_values[target]
    let value_max_value = normalizationJsonInfo.max_values[target]
    
    value = (normalizedValue * (value_max_value - value_min_value)) + value_min_value;

    return Math.round(value)
  }

document.querySelector('.calculate').addEventListener('click', async () => {

  if (allow == 1){
    
  allow = 0  
  MedianHouseValue.classList.add("hidden")
  loadingImage.classList.remove("hidden")

  const values = document.querySelectorAll('.input-container input');

  const saveValue = values[7].value
  values[7].value = formatValue(values[7].value)

  if (!values[7].value){
    allow = 1 
    loadingImage.classList.add("hidden")
    return window.alert("The Median income cannot have decimal places");
  }

  console.log(values[7].value)

  const normalizeValues = await normalizeInputs(values); 
  values[7].value = saveValue

  if(!normalizeValues.status){
    allow = 1 
    loadingImage.classList.add("hidden")
    return window.alert(normalizeValues.message);
  }

  const algorithm = document.querySelector(".algorithm-input").value
  let deNormalizePrediction = 0
  

  if (algorithm == "neural_network"){
    let normalizePrediction = await predict(normalizeValues.data, "./models/deep_learning_model.onnx", "input")
    deNormalizePrediction = await deNormalizeValue(normalizePrediction.output.data[0] , "median_house_value")
  }
  else {
    let normalizePrediction = await predict(normalizeValues.data, "./models/light_v2_bagging_model.onnx", "float_input")
    deNormalizePrediction = await deNormalizeValue(normalizePrediction.variable.cpuData[0] , "median_house_value")
  }

  loadingImage.classList.add("hidden")
  MedianHouseValue.classList.remove("hidden")
  document.querySelector("#median_house_value_value").innerHTML = " $" + Math.abs(deNormalizePrediction)
  allow = 1
  
}
});

document.querySelector(".algorithm-input").addEventListener("input", (e)=>{

  const warningText = document.querySelector(".warning-text")

   if(e.target.value == "bagging"){
    warningText.classList.remove("hidden")
   }
   else {
    warningText.classList.add("hidden")
   }

})