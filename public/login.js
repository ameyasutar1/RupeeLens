const loginTab = document.querySelector("#loginTab");
const registerTab = document.querySelector("#registerTab");
const loginForm = document.querySelector("#loginForm");
const registerForm = document.querySelector("#registerForm");

function showForm(kind) {
  const login = kind === "login";
  loginForm.hidden = !login;
  registerForm.hidden = login;
  loginTab.classList.toggle("active", login);
  registerTab.classList.toggle("active", !login);
}

async function submitAuth(form, endpoint, errorElement) {
  errorElement.textContent = "";
  const button = form.querySelector("button[type=submit]");
  button.disabled = true;
  try {
    const response = await fetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(Object.fromEntries(new FormData(form))),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "Authentication failed.");
    window.location.replace("/");
  } catch (error) {
    errorElement.textContent = error.message;
  } finally {
    button.disabled = false;
  }
}

loginTab.addEventListener("click", () => showForm("login"));
registerTab.addEventListener("click", () => showForm("register"));
loginForm.addEventListener("submit", event => {
  event.preventDefault();
  submitAuth(loginForm, "/api/auth/login", document.querySelector("#loginError"));
});
registerForm.addEventListener("submit", event => {
  event.preventDefault();
  submitAuth(registerForm, "/api/auth/register", document.querySelector("#registerError"));
});

fetch("/api/auth/status")
  .then(response => response.json())
  .then(status => {
    registerTab.hidden = !status.signup_available;
    document.querySelector("#signupCodeField").hidden = !status.signup_code_required;
  })
  .catch(() => {
    registerTab.hidden = true;
  });
