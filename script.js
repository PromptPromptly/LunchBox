const audioFileInput = document.getElementById("audioFile");
const fileNameText = document.getElementById("fileName");
const separateBtn = document.getElementById("separateBtn");
const loadingSection = document.getElementById("loadingSection");
const stemsSection = document.getElementById("stemsSection");
const stemButtons = document.querySelectorAll(".stem-btn");

let uploadedFile = null;

audioFileInput.addEventListener("change", (event) => {
  const [file] = event.target.files;
  uploadedFile = file || null;

  if (uploadedFile) {
    fileNameText.textContent = `Selected: ${uploadedFile.name}`;
    stemsSection.classList.add("hidden");
  } else {
    fileNameText.textContent = "No file selected";
  }
});

separateBtn.addEventListener("click", () => {
  if (!uploadedFile) {
    alert("Please upload an audio file first.");
    return;
  }

  separateBtn.disabled = true;
  stemsSection.classList.add("hidden");
  loadingSection.classList.remove("hidden");

  window.setTimeout(() => {
    loadingSection.classList.add("hidden");
    stemsSection.classList.remove("hidden");
    separateBtn.disabled = false;
  }, 3000);
});

stemButtons.forEach((button) => {
  button.addEventListener("click", () => {
    if (!uploadedFile) {
      alert("Please upload an audio file first.");
      return;
    }

    const url = URL.createObjectURL(uploadedFile);
    const link = document.createElement("a");
    link.href = url;

    const stemName = button.dataset.stem || "stem";
    const safeName = uploadedFile.name.replace(/\.[^/.]+$/, "");
    link.download = `${safeName}-${stemName}${getFileExtension(uploadedFile.name)}`;

    document.body.appendChild(link);
    link.click();
    link.remove();
    URL.revokeObjectURL(url);
  });
});

function getFileExtension(filename) {
  const dotIndex = filename.lastIndexOf(".");
  return dotIndex >= 0 ? filename.slice(dotIndex) : "";
}
