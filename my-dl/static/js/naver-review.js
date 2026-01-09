$(document).ready(function () {

    // Character count
    $('#review-content').on('input', function () {
        let content = $(this).val();
        let len = content.length;
        $('.char-count').text(len + ' / 200');

        if (len > 200) {
            $('.char-count').css('color', 'red');
        } else {
            $('.char-count').css('color', '#999');
        }
    });

    // Analyze click
    $('#btn-analyze').click(function () {
        let content = $('#review-content').val().trim();

        if (!content) {
            alert("리뷰 내용을 입력해주세요.");
            return;
        }

        // UI Reset & Loading
        let $btn = $(this);
        let originalText = $btn.text();
        $btn.prop('disabled', true).text('분석 중...');
        $('#result-area').addClass('hidden');

        $.ajax({
            url: '/review/naver',
            type: 'POST',
            contentType: 'application/json',
            data: JSON.stringify({ content: content }),
            success: function (response) {
                // Success
                let result = response.result; // "긍정" or "부정"
                let prob = response.probability; // score * 100

                let $resultText = $('#result-text');
                let $scoreBar = $('#score-bar');

                $resultText.text(result);

                if (result === '긍정') {
                    $resultText.removeClass('negative').css('color', '#03C75A');
                    $scoreBar.removeClass('negative').css('background-color', '#03C75A');
                } else {
                    $resultText.addClass('negative').css('color', '#FF5959');
                    $scoreBar.addClass('negative').css('background-color', '#FF5959');
                }

                $('#result-desc').html(`이 리뷰는 <strong>${prob}%</strong> 확률로 <strong>${result}</strong>적인 리뷰입니다.`);

                $('#result-area').removeClass('hidden');

                // Animate bar
                setTimeout(function () {
                    $scoreBar.css('width', prob + '%');
                }, 100);

            },
            error: function (xhr, status, error) {
                // Error
                let msg = "오류가 발생했습니다.";
                if (xhr.responseJSON && xhr.responseJSON.error) {
                    msg = xhr.responseJSON.error;
                }
                alert(msg);
            },
            complete: function () {
                // Restore button
                $btn.prop('disabled', false).text(originalText);
            }
        });
    });
});
