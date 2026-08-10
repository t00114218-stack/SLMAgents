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

// Category filtering and Search for 20+ upcoming agents
let activeCategory = 'all';

function filterCategory(btn, category) {
  // Update active tab styling
  const tabContainer = btn.parentElement;
  const buttons = tabContainer.querySelectorAll('.category-tab-btn');
  buttons.forEach(b => b.classList.remove('active'));
  btn.classList.add('active');
  
  activeCategory = category;
  applyFilters();
}

function applyFilters() {
  const searchInput = document.getElementById('agent-search');
  if (!searchInput) return;
  
  const searchQuery = searchInput.value.toLowerCase();
  const cards = document.querySelectorAll('.upcoming-card');
  
  cards.forEach(card => {
    const title = card.querySelector('h3').innerText.toLowerCase();
    const description = card.querySelector('p').innerText.toLowerCase();
    const cardCategory = card.getAttribute('data-category');
    
    const matchesSearch = title.includes(searchQuery) || description.includes(searchQuery);
    const matchesCategory = activeCategory === 'all' || cardCategory === activeCategory;
    
    if (matchesSearch && matchesCategory) {
      card.style.display = 'block';
    } else {
      card.style.display = 'none';
    }
  });
}

