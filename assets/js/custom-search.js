document.addEventListener('DOMContentLoaded', function() {
  // 외부 링크를 새 탭에서 열기
  document.querySelectorAll('.page__content a[href^="http"]').forEach(link => {
    link.setAttribute('target', '_blank');
    link.setAttribute('rel', 'noopener noreferrer');
  });

  // 필요한 요소들 생성
  const searchOverlay = document.createElement('div');
  searchOverlay.className = 'search-overlay';
  document.body.appendChild(searchOverlay);

  // 검색 토글 버튼에 이벤트 리스너 추가
  const searchToggle = document.querySelector('.search-toggle');
  if (searchToggle) {
    searchToggle.addEventListener('click', function(e) {
      e.preventDefault();
      document.body.classList.add('search-overlay-active');
      searchOverlay.style.display = 'block';
      const searchContent = document.querySelector('.search-content');
      if (searchContent) {
        searchContent.style.display = 'block';
        const searchInput = searchContent.querySelector('.search-input');
        if (searchInput) {
          searchInput.focus();
        }
      }
    });
  }

  // 오버레이 클릭시 검색창 닫기
  searchOverlay.addEventListener('click', function(e) {
    if (e.target === searchOverlay) {
      closeSearch();
    }
  });

  // ESC 키 누를 때 검색창 닫기
  document.addEventListener('keydown', function(e) {
    if (e.key === 'Escape') {
      closeSearch();
    }
  });

  function closeSearch() {
    document.body.classList.remove('search-overlay-active');
    searchOverlay.style.display = 'none';
    const searchContent = document.querySelector('.search-content');
    if (searchContent) {
      searchContent.style.display = 'none';
    }
  }

  // 다크모드 토글 버튼 생성 및 추가
  const themeToggle = document.createElement('button');
  themeToggle.className = 'theme-toggle';
  themeToggle.innerHTML = '<i class="fas fa-moon"></i>';
  document.body.appendChild(themeToggle);

  // 저장된 테마 적용
  const savedTheme = localStorage.getItem('theme');
  if (savedTheme) {
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeIcon(savedTheme);
  }

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
    icon.className = theme === 'dark' ? 'fas fa-sun' : 'fas fa-moon';
  }

  // 언어 전환 기능
  const languageToggle = document.createElement('button');
  languageToggle.className = 'language-switch-btn';
  languageToggle.innerHTML = '<span>🌐</span><span class="current-lang">KR</span>';
  document.body.appendChild(languageToggle);

  // 저장된 언어 설정 적용
  const savedLanguage = localStorage.getItem('language') || 'ko';
  document.documentElement.setAttribute('data-language', savedLanguage);
  updateLanguageContent(savedLanguage);
  updateLanguageButton(savedLanguage);

  // 언어 전환 이벤트
  languageToggle.addEventListener('click', function() {
    const currentLang = document.documentElement.getAttribute('data-language');
    const newLang = currentLang === 'en' ? 'ko' : 'en';
    
    document.documentElement.setAttribute('data-language', newLang);
    localStorage.setItem('language', newLang);
    updateLanguageContent(newLang);
    updateLanguageButton(newLang);
  });

  function updateLanguageButton(lang) {
    const currentLangSpan = languageToggle.querySelector('.current-lang');
    currentLangSpan.textContent = lang.toUpperCase();
  }

  function updateLanguageContent(lang) {
    const koreanContent = document.querySelectorAll('.korean-content');
    const englishContent = document.querySelectorAll('.english-content');

    if (lang === 'ko') {
      koreanContent.forEach(el => el.style.display = 'block');
      englishContent.forEach(el => el.style.display = 'none');
    } else {
      koreanContent.forEach(el => el.style.display = 'none');
      englishContent.forEach(el => el.style.display = 'block');
    }
  }
}); 