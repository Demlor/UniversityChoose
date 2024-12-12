// Đảm bảo DOM được tải đầy đủ trước khi chạy script
document.addEventListener('DOMContentLoaded', function () {
    // Gắn hàm vào window để có thể sử dụng với onclick
    window.setAnswer = function (inputId, value, button) {
        const inputElement = document.getElementById(inputId);
        if (!inputElement) {
            console.error(`Không tìm thấy phần tử với ID: ${inputId}`);
            return;
        }

        inputElement.value = value;

        // Đổi màu cho nút đã chọn
        const buttons = button.parentElement.querySelectorAll('button');
        buttons.forEach(btn => btn.classList.remove('selected'));
        button.classList.add('selected');
    };

    window.toggleFields = function (option, inputId, button) {
        const fieldsContainer = document.getElementById('fields-container_3');
        const inputElement = document.getElementById(inputId);

        if (!fieldsContainer || !inputElement) {
            console.error('Không tìm thấy container hoặc input liên quan.');
            return;
        }

        const buttons = button.parentElement.querySelectorAll('button');
        buttons.forEach(btn => btn.classList.remove('selected'));
        button.classList.add('selected');

        if (option === 'yes') {
            fieldsContainer.style.display = 'block';
            inputElement.value = 'Có';
        } else {
            fieldsContainer.style.display = 'none';
            inputElement.value = 'Không';
        }
    };

    // Thay thế onclick bằng addEventListener
    const answerButtons = document.querySelectorAll('.answer-button');
    answerButtons.forEach(button => {
        button.addEventListener('click', function () {
            const inputId = button.dataset.input;
            const value = button.dataset.value;
            setAnswer(inputId, value, button);
        });
    });

    const toggleFieldButtons = document.querySelectorAll('.toggle-field-button');
    toggleFieldButtons.forEach(button => {
        button.addEventListener('click', function () {
            const inputId = button.dataset.input;
            const value = button.dataset.value;
            toggleFields(value, inputId, button);
        });
    });

    // Xử lý khi form được gửi
    const careerForm = document.getElementById('careerForm');
    if (careerForm) {
        careerForm.addEventListener('submit', function (event) {
            event.preventDefault(); // Ngăn hành động mặc định của form

            // Lấy dữ liệu từ form
            const data = {
                family_advice: document.getElementById('family_advice')?.value || '',
                financial_influence: document.getElementById('financial_influence')?.value || '',
                family_industry: document.getElementById('family_industry')?.value || '',
                family_industry_select: document.getElementById('family_industry_select')?.value || '',
                mbti: document.getElementById('mbti')?.value || '',
                subjects: Array.from(document.querySelectorAll('input[name="subjects"]:checked')).map(input => input.value),
                strengths: Array.from(document.querySelectorAll('input[name="strengths"]:checked')).map(input => input.value),
                interests: Array.from(document.querySelectorAll('input[name="interests"]:checked')).map(input => input.value)
            };

            // Gửi dữ liệu tới backend
            fetch('/submit', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(data)
            })
                .then(response => response.json())
                .then(responseData => {
                    console.log('Kết quả gửi form:', responseData);
                })
                .catch(error => {
                    console.error('Lỗi khi gửi form:', error);
                });
        });
    }

    // Hàm xác thực form
    function validateForm() {
        const mbtiSelect = document.getElementById('mbti');
        const subjectsInputs = document.querySelectorAll('input[name="subjects"]');
        const strengthsInputs = document.querySelectorAll('input[name="strengths"]');
        const interestsInputs = document.querySelectorAll('input[name="interests"]');
        const submitButton = document.getElementById('submit');

        if (!mbtiSelect || !submitButton) {
            console.error('Không tìm thấy phần tử cần thiết để xác thực.');
            return;
        }

        const mbtiSelected = mbtiSelect.value !== '';
        const subjectsChecked = Array.from(subjectsInputs).some(input => input.checked);
        const strengthsChecked = Array.from(strengthsInputs).some(input => input.checked);
        const interestsChecked = Array.from(interestsInputs).some(input => input.checked);

        submitButton.disabled = !(mbtiSelected && subjectsChecked && strengthsChecked && interestsChecked);
    }

    // Lắng nghe sự kiện thay đổi để xác thực form
    const mbtiSelect = document.getElementById('mbti');
    const subjectsInputs = document.querySelectorAll('input[name="subjects"]');
    const strengthsInputs = document.querySelectorAll('input[name="strengths"]');
    const interestsInputs = document.querySelectorAll('input[name="interests"]');

    if (mbtiSelect) {
        mbtiSelect.addEventListener('change', validateForm);
    }
    subjectsInputs.forEach(input => input.addEventListener('change', validateForm));
    strengthsInputs.forEach(input => input.addEventListener('change', validateForm));
    interestsInputs.forEach(input => input.addEventListener('change', validateForm));
});
