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
    // var url = 'http://127.0.0.1:5000/'
    var url = 'https://letterrecognitionapi.azurewebsites.net/'
    const response = await fetch(url, {
        method: 'POST',
        body: JSON.stringify({ "image": image_b64 }),
        headers: {
            'Content-Type': 'application/json'
        }
    });
    const json_response = await response.json();

    // Response
    var predicted = json_response["predicted"]
    var confidence = json_response["confidence"]
    var all_letters = json_response["all"]

    // Clean and sort
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

    extra_info = ""
    if (all_letters.length === 0) {
        extra_info += "<b>Confidence</b>: 100%<br>"
    }
    else {
        extra_info += "<b>Confidence</b>: " + Math.round((confidence * 100)).toString() + "%<br>"
        extra_info += "<b>Other possibilities</b>:<br>"
        for (var i = 0; i < all_letters.length; i++) {
            var letter = all_letters[i]["letter"];
            var probability = all_letters[i]["probability"];
            extra_info += letter + ": " + Math.round((probability * 100)).toString() + "%<br>";
        }
    }

    Swal.fire({
        title: "I think your letter is " + predicted + ".",
        confirmButtonColor: '#ff3b3f',
        confirmButtonText: 'Try again!',
        showDenyButton: true,
        denyButtonColor: '#a9a9a9',
        denyButtonText: 'Show details',
        background: '#efefef'
    }).then((result) => {
        if (result.isConfirmed) {
            window.location.reload()
        } else if (result.isDenied) {
            Swal.fire({
                html: extra_info,
                confirmButtonColor: '#ff3b3f',
                confirmButtonText: 'Try again!'
            }).then((result) => {
                if (result.isConfirmed) {
                    window.location.reload()
                }
            })
        }
    })
}