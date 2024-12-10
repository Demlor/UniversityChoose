document.addEventListener("DOMContentLoaded", () => {
    console.log("JavaScript đã được tải thành công!");
    const form = document.getElementById("careerForm");

    form.addEventListener("submit", (event) => {
        event.preventDefault(); // Ngăn gửi form mặc định
        alert("Form đã gửi thành công!");
    });
});
