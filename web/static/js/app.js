// ===========================
// TẠO SỰ KIỆN KÉO-THẢ VÀ LẤY URL TỪ KÉO-THẢ
// ===========================

const dropArea = document.getElementById('drop-area');
const inputField = document.getElementById('product-url');
const submitButton = document.getElementById('submit-btn');
const progressBarBERT = document.getElementById('bert-progress-bar'); 
const progressBarLSTM = document.getElementById('lstm-progress-bar'); 
const progressDescriptionBERT = document.getElementById('bert-progress-description'); 
const progressDescriptionLSTM = document.getElementById('lstm-progress-description');
const bertResultButton = document.getElementById('view-bert-results');
const lstmResultButton = document.getElementById('view-lstm-results');

// Ngăn hành vi mặc định khi kéo URL vào vùng thả
dropArea.addEventListener('dragover', (e) => e.preventDefault());
dropArea.addEventListener('drop', (e) => {
    e.preventDefault();
    const url = e.dataTransfer.getData('text');
    if (url.startsWith('http')) {
        inputField.value = url;
    } else {
        alert("Vui lòng kéo thả một URL hợp lệ!");
    }
});

// // ===========================
// // CHỌN MÔ HÌNH PHÂN TÍCH CẢM XÚC
// // ===========================

// modelButtons.forEach(button => {
//     button.addEventListener('click', () => {
//         // Toggle trạng thái "selected"
//         if (button.classList.contains('selected')) {
//             button.classList.remove('selected');
//             selectedModel = null;
//         } else {
//             // Đảm bảo chỉ có một nút được chọn
//             modelButtons.forEach(btn => btn.classList.remove('selected'));
//             button.classList.add('selected');
//             selectedModel = button.getAttribute('id');
//         }
//     });
// });

// ===========================
// CẬP NHẬT THANH TIẾN TRÌNH
// ===========================
function updateProgressBar(progressBar, progressDescription, percentage, message) {
    progressBar.style.width = `${percentage}%`;
    progressDescription.textContent = message;
}

// ===========================
// GỬI YÊU CẦU TỚI BACKEND
// ===========================
submitButton.addEventListener('click', (e) => {
    e.preventDefault();
    const url = inputField.value.trim();

    if (!url) {
        alert("Vui lòng nhập URL!");
        return;
    }

    // Khởi động thanh tiến trình
    updateProgressBar(progressBarBERT, progressDescriptionBERT, 0, "Đang bắt đầu...");
    updateProgressBar(progressBarLSTM, progressDescriptionLSTM, 0, "Đang bắt đầu...");

    $.ajax({
        url: '/crawl',
        method: 'POST',
        data: JSON.stringify({ url }),
        contentType: 'application/json',
        beforeSend: function () {
            updateProgressBar(progressBarBERT, progressDescriptionBERT, 20, "Đang lấy dữ liệu từ URL...");
            updateProgressBar(progressBarLSTM, progressDescriptionLSTM, 20, "Đang lấy dữ liệu từ URL...");
        },
        success: function () {
            updateProgressBar(progressBarBERT, progressDescriptionBERT, 50, "Dữ liệu được lấy, phân tích BERT...");
            updateProgressBar(progressBarLSTM, progressDescriptionLSTM, 50, "Dữ liệu được lấy, phân tích LSTM...");
            analyzeComments('bert', progressBarBERT, progressDescriptionBERT, bertResultButton);
            analyzeComments('lstm', progressBarLSTM, progressDescriptionLSTM, lstmResultButton);
        },
        error: function (error) {
            handleAjaxError(progressBarBERT, progressDescriptionBERT, error);
            handleAjaxError(progressBarLSTM, progressDescriptionLSTM, error);
        }
    });
});

function analyzeComments(model, progressBar, progressDescription, resultButton) {
    $.ajax({
        url: '/analyze',
        method: 'POST',
        data: JSON.stringify({ model }),
        contentType: 'application/json',
        beforeSend: function () {
            updateProgressBar(progressBar, progressDescription, 70, `Phân tích bằng ${model.toUpperCase()}...`);
        },
        success: function () {
            updateProgressBar(progressBar, progressDescription, 100, `Hoàn tất phân tích bằng ${model.toUpperCase()}!`);
            resultButton.disabled = false;
        },
        error: function (error) {
            handleAjaxError(progressBar, progressDescription, error);
        }
    });
}

function handleAjaxError(progressBar, progressDescription, error) {
    updateProgressBar(progressBar, progressDescription, 0, "Lỗi: " + (error.responseJSON?.error || "Không rõ lỗi"));
}

// ===========================
// HÀM CHẠY MÔ HÌNH PHÂN TÍCH
// ===========================

// function analyzeComments(model, progressBar, progressDescription, resultButton) {
//     $.ajax({
//         url: '/analyze',
//         method: 'POST',
//         data: { model: model }, // Gửi mô hình
//         beforeSend: function() {
//             updateProgressBar(progressBar, progressDescription, 70, `Đang phân tích cảm xúc bằng ${model.toUpperCase()}...`);
//         },
//         success: function(response) {
//             updateProgressBar(progressBar, progressDescription, 100, `Phân tích hoàn tất bằng ${model.toUpperCase()}!`);
//             resultButton.disabled = false; // Bật nút xem kết quả
//         },
//         error: function(error) {
//             alert(`Đã xảy ra lỗi khi phân tích cảm xúc bằng ${model.toUpperCase()}: ` + (error.responseJSON ? error.responseJSON.error : "Không rõ lỗi"));
//             updateProgressBar(progressBar, progressDescription, 0, `Lỗi khi phân tích cảm xúc bằng ${model.toUpperCase()}.`);
//         }
//     });
// }

// ===========================
// XEM KẾT QUẢ MÔ HÌNH
// ===========================

bertResultButton.addEventListener('click', (event) => {
    // event.preventDefault();  // Ngừng hành động mặc định (chuyển trang)
    
    if (bertResultButton.disabled) {
        alert("Phân tích bằng BERT chưa hoàn tất. Vui lòng chờ!");
    } else {
        // Mở cửa sổ pop-up với trang kết quả BERT
        window.open('/result?model=bert', 'resultBERT', 'width=800,height=600');
    }
});

lstmResultButton.addEventListener('click', (event) => {
    // event.preventDefault();  // Ngừng hành động mặc định (chuyển trang)

    if (lstmResultButton.disabled) {
        alert("Phân tích bằng LSTM chưa hoàn tất. Vui lòng chờ!");
    } else {
        // Mở cửa sổ pop-up với trang kết quả LSTM
        window.open('/result?model=lstm', 'resultLSTM', 'width=800,height=600');
    }
});
