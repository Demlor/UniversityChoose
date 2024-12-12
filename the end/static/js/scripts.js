document.addEventListener('DOMContentLoaded', function () {

    // Hàm set giá trị cho input ẩn khi người dùng chọn câu trả lời
    function setAnswer(inputId, value, button) {
        // Gán giá trị vào input ẩn
        document.getElementById(inputId).value = value;

        // Đổi màu cho nút đã chọn
        var buttons = button.parentElement.getElementsByTagName('button');
        for (var btn of buttons) {
            btn.classList.remove('selected'); // Xóa lớp 'selected' khỏi tất cả nút
        }
        button.classList.add('selected'); // Thêm lớp 'selected' cho nút đã chọn
    }

    // Hàm xử lý khi người dùng chọn "Có" hoặc "Không" trong câu hỏi về "Gia đình"
    function toggleFields(option, inputId, button) {
        var fieldsContainer = document.getElementById('fields-container_3'); // Lấy container của dropdown "Lĩnh vực gia đình"
        var familyIndustryInput = document.getElementById(inputId); // Lấy input ẩn cho "Gia đình có ngành nghề không?"

        // Thêm console log để kiểm tra khi hàm được gọi
        console.log('toggleFields called with option: ' + option);

        var buttons = button.parentElement.getElementsByTagName('button');
        for (var btn of buttons) {
            btn.classList.remove('selected'); // Xóa lớp 'selected' khỏi tất cả nút
        }
        button.classList.add('selected'); // Thêm lớp 'selected' cho nút đã chọn

        // Hiển thị hoặc ẩn dropdown dựa trên lựa chọn "Có" hoặc "Không"
        if (option === 'yes') {
            console.log('Showing dropdown');
            fieldsContainer.style.display = 'block'; // Hiển thị dropdown
            familyIndustryInput.value = 'Có'; // Cập nhật giá trị của input ẩn thành "Có"
        } else {
            console.log('Hiding dropdown');
            fieldsContainer.style.display = 'none'; // Ẩn dropdown
            familyIndustryInput.value = 'Không'; // Cập nhật giá trị của input ẩn thành "Không"
        }
    }

    // Xử lý khi form được gửi
    document.getElementById('careerForm').addEventListener('submit', function (event) {
        event.preventDefault();  // Ngừng hành động gửi form mặc định

        // Lấy tất cả dữ liệu từ form
        const data = {
            family_advice: document.getElementById('family_advice').value,
            financial_influence: document.getElementById('financial_influence').value,
            family_industry: document.getElementById('family_industry').value,
            family_industry_select_3: document.getElementById('family_industry_select_3') ? document.getElementById('family_industry_select_3').value : '',
            mbti: document.getElementById('mbti').value,
            subjects: Array.from(document.querySelectorAll('input[name="subjects"]:checked')).map(input => input.value),
            strengths: Array.from(document.querySelectorAll('input[name="strengths"]:checked')).map(input => input.value),
            interests: Array.from(document.querySelectorAll('input[name="interests"]:checked')).map(input => input.value)
        };

        // Gửi dữ liệu tới backend qua fetch API
        fetch('/submit', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(data) // Chuyển dữ liệu thành JSON
        })
        .then(response => response.json())
        .then(data => {
            console.log('Success:', data); // Xử lý phản hồi từ backend
        })
        .catch(error => {
            console.error('Error:', error); // Xử lý lỗi khi gửi dữ liệu
        });
    });

    // Kiểm tra tính hợp lệ của form khi thay đổi các trường
    const mbtiSelect = document.getElementById("mbti");
    const subjectsInputs = document.querySelectorAll('input[name="subjects"]');
    const strengthsInputs = document.querySelectorAll('input[name="strengths"]');
    const interestsInputs = document.querySelectorAll('input[name="interests"]');
    const submitButton = document.getElementById("submit");

    // Hàm xác thực form
    function validateForm() {
        const mbtiSelected = mbtiSelect.value !== "";
        const subjectsChecked = Array.from(subjectsInputs).some(input => input.checked);
        const strengthsChecked = Array.from(strengthsInputs).some(input => input.checked);
        const interestsChecked = Array.from(interestsInputs).some(input => input.checked);

        submitButton.disabled = !(mbtiSelected && subjectsChecked && strengthsChecked && interestsChecked);
    }

    // Lắng nghe các sự kiện thay đổi trong form
    mbtiSelect.addEventListener("change", validateForm);
    subjectsInputs.forEach(input => input.addEventListener("change", validateForm));
    strengthsInputs.forEach(input => input.addEventListener("change", validateForm));
    interestsInputs.forEach(input => input.addEventListener("change", validateForm));

    // Lấy kết quả ngành nghề từ backend (tùy chọn)
    async function fetchResults(isLoadMore = false) {
        const mbti = mbtiSelect.value;
        const subjects = Array.from(subjectsInputs).filter(input => input.checked).map(input => input.value);
        const strengths = Array.from(strengthsInputs).filter(input => input.checked).map(input => input.value);
        const interests = Array.from(interestsInputs).filter(input => input.checked).map(input => input.value);

        // Gửi dữ liệu đến backend (cần định tuyến API backend tại /path/to/api)
        const recommendations = await fetch('/path/to/api', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ mbti, subjects, strengths, interests })
        }).then(response => response.json());

        const resultsDiv = document.getElementById("results");

        if (!isLoadMore) {
            resultsDiv.innerHTML = "<h3>Gợi ý ngành nghề:</h3>";
        }

        if (recommendations.length > 0) {
            recommendations.forEach(rec => {
                resultsDiv.innerHTML += `<p>${rec.career} - Độ phù hợp: ${rec.score}</p>`;
            });

            if (recommendations.length === 5) {
                loadMoreButton.style.display = "block"; // Hiển thị nút "Xem thêm"
            } else {
                loadMoreButton.style.display = "none"; // Ẩn nút "Xem thêm"
            }
        } else {
            loadMoreButton.style.display = "none"; // Ẩn nút "Xem thêm"
        }
    }

    // Lắng nghe sự kiện nhấn nút "Gửi"
    submitButton.addEventListener("click", () => {
        page = 1;
        fetchResults(false);
    });

    // Lắng nghe sự kiện nhấn nút "Xem thêm" để tải thêm kết quả
    const loadMoreButton = document.getElementById("load-more");
    loadMoreButton.addEventListener("click", () => {
        fetchResults(true);
    });
});
