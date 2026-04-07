// ── Navbar scroll effect ──
const navbar = document.getElementById('navbar');
window.addEventListener('scroll', () => {
  navbar.classList.toggle('scrolled', window.scrollY > 40);
});

// ── Hamburger menu ──
document.getElementById('hamburger').addEventListener('click', () => {
  document.querySelector('.nav-links').classList.toggle('open');
});
document.querySelectorAll('.nav-links a').forEach(a =>
  a.addEventListener('click', () => document.querySelector('.nav-links').classList.remove('open'))
);

// ── Animated counters ──
function animateCounter(el) {
  const target = +el.dataset.target;
  const step = Math.ceil(target / 60);
  let current = 0;
  const timer = setInterval(() => {
    current = Math.min(current + step, target);
    el.textContent = current.toLocaleString();
    if (current >= target) clearInterval(timer);
  }, 25);
}

// ── Intersection Observer ──
const observer = new IntersectionObserver((entries) => {
  entries.forEach(entry => {
    if (!entry.isIntersecting) return;

    // Counter animation
    if (entry.target.classList.contains('stat-num')) {
      animateCounter(entry.target);
      observer.unobserve(entry.target);
    }

    // Timeline animation
    if (entry.target.classList.contains('timeline-item')) {
      entry.target.classList.add('visible');
      observer.unobserve(entry.target);
    }

    // Bar chart animation
    if (entry.target.classList.contains('bar-fill')) {
      entry.target.style.width = entry.target.style.getPropertyValue('--w');
      observer.unobserve(entry.target);
    }
  });
}, { threshold: 0.2 });

document.querySelectorAll('.stat-num, .timeline-item, .bar-fill').forEach(el => observer.observe(el));

// ── Model card flip ──
function toggleCard(card) {
  card.classList.toggle('flipped');
}

// ── Tab switching ──
function switchTab(btn, id) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
  btn.classList.add('active');
  document.getElementById('tab-' + id).classList.add('active');
}

// ── Active nav link highlight on scroll ──
const sections = document.querySelectorAll('section[id], header[id]');
window.addEventListener('scroll', () => {
  let current = '';
  sections.forEach(s => {
    if (window.scrollY >= s.offsetTop - 100) current = s.id;
  });
  document.querySelectorAll('.nav-links a').forEach(a => {
    a.style.color = a.getAttribute('href') === '#' + current ? 'var(--accent)' : '';
  });
});
