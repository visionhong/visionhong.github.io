document.addEventListener('DOMContentLoaded', function() {
  const gridWrapper = document.querySelector('.grid__wrapper');
  const loadMoreBtn = document.createElement('div');
  loadMoreBtn.className = 'load-more';
  loadMoreBtn.innerHTML = `
    <button class="btn--load-more">
      <span>Load More</span>
      <svg class="icon" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
        <path d="M12 4V2M12 22v-2M6.34 6.34L4.93 4.93M19.07 19.07l-1.41-1.41M4 12H2M22 12h-2M6.34 17.66l-1.41 1.41M19.07 4.93l-1.41 1.41" 
          stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
      </svg>
    </button>
  `;

  let currentPage = 1;
  const postsPerPage = 9;
  const posts = Array.from(document.querySelectorAll('.grid__item'));
  const totalPages = Math.ceil(posts.length / postsPerPage);

  // 초기 포스트만 표시
  posts.forEach((post, index) => {
    if (index >= postsPerPage) {
      post.style.display = 'none';
    }
  });

  // Load More 버튼이 필요한 경우에만 추가
  if (posts.length > postsPerPage) {
    gridWrapper.after(loadMoreBtn);
  }

  loadMoreBtn.addEventListener('click', function() {
    const button = loadMoreBtn.querySelector('.btn--load-more');
    button.classList.add('loading');

    // 다음 페이지의 포스트 표시
    setTimeout(() => {
      const start = currentPage * postsPerPage;
      const end = start + postsPerPage;
      
      posts.slice(start, end).forEach(post => {
        post.style.display = '';
        post.style.animation = 'fadeIn 0.5s ease forwards';
      });

      currentPage++;

      // 마지막 페이지인 경우 버튼 숨김
      if (currentPage >= totalPages) {
        loadMoreBtn.style.display = 'none';
      }

      button.classList.remove('loading');
    }, 500); // 로딩 효과를 위한 지연
  });
});

// 페이드인 애니메이션
const style = document.createElement('style');
style.textContent = `
  @keyframes fadeIn {
    from {
      opacity: 0;
      transform: translateY(20px);
    }
    to {
      opacity: 1;
      transform: translateY(0);
    }
  }
`;
document.head.appendChild(style); 