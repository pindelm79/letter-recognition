// Canvas
window.onload = function () {
    var canvas = new fabric.Canvas('sheet');
    canvas.isDrawingMode = true;
    canvas.setBackgroundColor('#a9a9a9', canvas.renderAll.bind(canvas));
    canvas.freeDrawingBrush.width = 20;
    canvas.freeDrawingBrush.color = "#ffffff";
}

// Image sending
const sendImage = async () => {
    // Sending
    var canvas = document.getElementById("sheet");
    var image_b64 = canvas.toDataURL().replace("data:image/png;base64,", "");
    const response = await fetch('https://letterrecognitionapi.azurewebsites.net/', {
        method: 'POST',
        body: JSON.stringify({ "image": image_b64 }),
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const json_response = await response.json();

    // Response
    predicted = json_response["predicted"];
    probabilities = json_response["probabilities"];
    var predicted_para = document.createElement("P");
    predicted_para.innerText = "Predicted letter: " + predicted;
    document.body.appendChild(predicted_para);

    // Cleanup
    delete probabilities[predicted];
    function clean(obj) {
        for (var propName in obj) {
            if (obj[propName] === null || obj[propName] === undefined || obj[propName] == 0) {
                delete obj[propName];
            }
        }
        return obj;
    }
    var probabilities_clean = clean(probabilities);

    var probabilities_para = document.createElement("P");
    probabilities_para.innerText = "Other possibilities: " + JSON.stringify(probabilities_clean);
    document.body.appendChild(probabilities_para);
}