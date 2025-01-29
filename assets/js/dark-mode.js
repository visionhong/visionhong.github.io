document.addEventListener('DOMContentLoaded', function() {
  // 다크모드 토글 버튼 생성 및 추가
  const themeToggle = document.createElement('button');
  themeToggle.className = 'theme-toggle';
  themeToggle.setAttribute('aria-label', '다크모드 토글');
  themeToggle.innerHTML = '<i class="fa-solid fa-circle-half-stroke"></i>';
  document.body.appendChild(themeToggle);

  // 저장된 테마 적용 (기본값을 dark로 설정)
  const savedTheme = localStorage.getItem('theme') || 'dark';
  document.documentElement.setAttribute('data-theme', savedTheme);
  updateThemeIcon(savedTheme);

  // 테마 토글 이벤트
  themeToggle.addEventListener('click', function() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeIcon(newTheme);
  });

  function updateThemeIcon(theme) {
    const icon = themeToggle.querySelector('i');
    // 아이콘은 동일하게 유지하고 CSS로 회전만 처리
    icon.style.transform = theme === 'dark' ? 'rotate(180deg)' : 'rotate(0deg)';
  }
}); 