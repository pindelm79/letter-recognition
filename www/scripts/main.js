// Canvas
window.onload = function () {
    var canvas = new fabric.Canvas('sheet');
    document.getElementById("sheet").canvas = canvas
    canvas.isDrawingMode = true;
    canvas.setBackgroundColor('#a9a9a9', canvas.renderAll.bind(canvas));
    canvas.freeDrawingBrush.width = 20;
    canvas.freeDrawingBrush.color = "#ffffff";
}

function clearCanvas() {
    var canvas = document.getElementById("sheet").canvas
    canvas.clear()
    canvas.setBackgroundColor('#a9a9a9', canvas.renderAll.bind(canvas));
}

// Image sending
const sendImage = async () => {
    // Connection indicator start
    var icon = document.getElementById("loading-icon");
    icon.classList.add("fa", "fa-spinner", "fa-spin");

    // Response sending and handling
    var canvas = document.getElementById("sheet");
    var image_b64 = canvas.toDataURL().replace("data:image/png;base64,", "");
    var url = 'http://127.0.0.1:5000/'
    // var url = 'https://letterrecognitionapi.azurewebsites.net/'
    const response = await fetch(url, {
        method: 'POST',
        body: JSON.stringify({ "image": image_b64 }),
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const json_response = await response.json();
    var predicted = json_response["predicted"]
    var confidence = json_response["confidence"]
    var all_letters = json_response["all"]

    // Cleaning and sorting "all"
    function GetSortOrder(prop) {
        return function (a, b) {
            if (a[prop] > b[prop]) {
                return -1;
            } else if (a[prop] < b[prop]) {
                return 1;
            }
            return 0;
        }
    }
    all_letters.sort(GetSortOrder("probability"));
    delete all_letters[0]
    all_letters = all_letters.filter((el) => {
        return parseFloat(el.probability.toFixed(2)) !== 0;
    })



    // Preparing display text
    main_info = ""
    extra_info = ""
    confidence_percent = Math.round(confidence * 100)
    if (confidence_percent <= 50) {
        main_info += "<b>Oops! I'm not sure what letter this is.<br>See 'Extra info' or try again.</b>"
        extra_info += "<b>Possible guesses:<br></b>"
        for (var i = 0; i < all_letters.length; i++) {
            var letter = all_letters[i]["letter"];
            var probability = all_letters[i]["probability"];
            extra_info += letter + ": " + Math.round(probability * 100).toString() + "%<br>";
        }
    }
    else {
        main_info += "I am " + confidence_percent.toString() + "% sure your letter is " + predicted + "."
        extra_info = "<b>Predicted letter</b>: " + predicted + "<br>"
        extra_info += "<b>Confidence</b>: " + confidence_percent.toString() + "%<br>"
        if (all_letters.length === 0) {
            extra_info += "No other possibilities."
        }
        else {
            extra_info += "<b>Other possibilities</b>:<br>"
            for (var i = 0; i < all_letters.length; i++) {
                var letter = all_letters[i]["letter"];
                var probability = all_letters[i]["probability"];
                extra_info += letter + ": " + Math.round(probability * 100).toString() + "%<br>";
            }
        }
    }

    // Connection indicator stop
    icon.classList.remove("fa", "fa-spinner", "fa-spin");

    Swal.fire({
        // title: main_info,
        html: main_info,
        confirmButtonColor: '#ff3b3f',
        confirmButtonText: 'Try again!',
        showDenyButton: true,
        denyButtonColor: '#a9a9a9',
        denyButtonText: 'Extra info',
        background: '#efefef'
    }).then((result) => {
        if (result.isConfirmed) {
            clearCanvas()
        } else if (result.isDenied) {
            Swal.fire({
                html: extra_info,
                confirmButtonColor: '#ff3b3f',
                confirmButtonText: 'Try again!'
            }).then((result) => {
                if (result.isConfirmed) {
                    clearCanvas()
                }
            })
        }
    })
}