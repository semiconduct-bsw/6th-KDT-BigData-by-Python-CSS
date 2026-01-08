$(document).ready(function () {
    $('#analyzeBtn').on('click', function () {
        var sl = $('#sepal_length').val();
        var sw = $('#sepal_width').val();
        var pl = $('#petal_length').val();
        var pw = $('#petal_width').val();

        $.ajax({
            url: '/ai/predict',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({
                // 백엔드에 있는 predict_iris 함수의 파라미터와 일치
                sepal_length: sl,
                sepal_width: sw,
                petal_length: pl,
                petal_width: pw
            }),
            success: function (response) {
                if (response.success) {
                    $('#result').addClass('active');
                    $('.result-icon').text('🌸');
                    $('.result-title').text('예측 완료');
                    $('#resultValue').text(response["예측된 클래스 종류"]);
                } else {
                    alert(response.message);
                }
            },
            error: function (xhr, status, error) {
                alert('예측 실패');
            }
        });
    });
});