// Replace this with your actual backend endpoint
const BACKEND_URL = "https://my-recolor-app.onrender.com";

let chosenColor = null;

function selectColor(color) {
  chosenColor = color;
  document.getElementById("selectedColor").innerText = color;
}

async function uploadAndRecolor() {
  const fileInput = document.getElementById("imageInput");
  const file = fileInput.files[0];
  const statusEl = document.getElementById("statusMessage");
  const resultImg = document.getElementById("resultImage");
  const loadingBar = document.getElementById("loadingBar");

  if (!file) {
    alert("Please select an image file first!");
    return;
  }
  if (!chosenColor) {
    alert("Please select a color first!");
    return;
  }

  // Clear previous status/message and image
  statusEl.innerText = "";
  resultImg.style.display = "none";   // Hide the old image
  loadingBar.style.display = "block"; // Show loading bar

  // Prepare form data
  const formData = new FormData();
  formData.append("image", file);
  formData.append("color", chosenColor);

  try {
    // Show the user we are "processing..."
    statusEl.innerText = "Processing... please wait.";

    // Send the request to your backend
    const response = await fetch(`${BACKEND_URL}/recolor`, {
      method: "POST",
      body: formData
    });

    if (!response.ok) {
      let errorMsg = "An error occurred.";
      try {
        const errorData = await response.json();
        if (errorData.error) errorMsg = errorData.error;
      } catch (_) {}
      statusEl.innerText = `Error: ${errorMsg}`;
      // Hide loading bar, keep the container blank
      loadingBar.style.display = "none";
      return;
    }

    // The response is a binary image
    const blob = await response.blob();
    const imageUrl = URL.createObjectURL(blob);

    // Hide the loading bar, show the image
    loadingBar.style.display = "none";
    resultImg.src = imageUrl;
    resultImg.style.display = "block";
    statusEl.innerText = "Done!";
  } catch (err) {
    console.error(err);
    statusEl.innerText = "Error: " + err.message;
    loadingBar.style.display = "none";
  }
}
