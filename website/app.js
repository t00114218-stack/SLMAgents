// Tab switching logic for code panels
function switchTab(btn, targetId) {
  // Get all buttons in the same container
  const tabContainer = btn.parentElement;
  const buttons = tabContainer.querySelectorAll('.code-tab-btn');
  buttons.forEach(b => b.classList.remove('active'));
  
  // Set current button to active
  btn.classList.add('active');
  
  // Hide all code contents in the same parent code panel
  const panel = tabContainer.parentElement;
  const contents = panel.querySelectorAll('.code-content');
  contents.forEach(c => c.style.display = 'none');
  
  // Show target content
  const target = panel.querySelector(`#${targetId}`);
  if (target) {
    target.style.display = 'block';
  }
}
