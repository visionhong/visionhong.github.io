document.addEventListener('DOMContentLoaded', () => {
  const postContainer = document.querySelector('.grid__wrapper');
  const paginationContainer = document.querySelector('.pagination'); // Pagination container

  if (!postContainer || !paginationContainer) {
    console.log('Post container (.grid__wrapper) or pagination container (.pagination) not found. Infinite scroll disabled for this page.');
    if (paginationContainer && !postContainer) { // If pagination exists but no posts, still hide pagination
        paginationContainer.style.display = 'none';
    }
    return;
  }

  let initialNextPath = ''; // Renamed for clarity
  let currentNextPath = ''; // Stores the path for the upcoming fetch
  let isLoading = false;
  let noMorePosts = false;
  let sentinel; // Declare sentinel here
  let observer; // Declare observer to unobserve if needed

  const getNextPathFromDocument = (doc = document) => {
    const nextPageLink = doc.querySelector('.pagination__link.next:not(.disabled)');
    return nextPageLink ? nextPageLink.getAttribute('href') : null;
  };

  initialNextPath = getNextPathFromDocument();

  if (!initialNextPath) {
    console.log('Initial next page path not found. Infinite scroll will not run.');
    paginationContainer.style.display = 'none'; // Hide pagination if no initial next link
    return;
  }
  
  currentNextPath = initialNextPath;
  paginationContainer.style.display = 'none'; // Hide pagination as we're using infinite scroll

  const loadMorePosts = async () => {
    if (isLoading || noMorePosts || !currentNextPath) return;

    isLoading = true;
    const spinner = document.createElement('div');
    spinner.classList.add('infinite-scroll-spinner');
    spinner.innerHTML = '<p>Loading...</p>';
    
    if (sentinel && sentinel.parentNode) {
        sentinel.parentNode.insertBefore(spinner, sentinel);
    } else {
        postContainer.parentNode.appendChild(spinner); // Fallback if sentinel isn't there
    }
    
    try {
      const response = await fetch(currentNextPath);
      if (!response.ok) {
        if (response.status === 404) {
          console.log('No more posts to load (404).');
          noMorePosts = true;
        } else {
          console.error('Error fetching next page:', response.statusText);
          // Consider not setting noMorePosts for other errors to allow retry on next scroll
        }
        if (noMorePosts && sentinel && observer) observer.unobserve(sentinel);
        return;
      }

      const text = await response.text();
      const parser = new DOMParser();
      const doc = parser.parseFromString(text, 'text/html');
      const newPostsWrapper = doc.querySelector('.grid__wrapper');

      if (newPostsWrapper && newPostsWrapper.children.length > 0) {
        Array.from(newPostsWrapper.children).forEach(postNode => {
          if (postNode.classList.contains('archive__item') || postNode.classList.contains('grid__item')) {
            postContainer.appendChild(postNode.cloneNode(true));
          }
        });
        
        const newFoundNextPath = getNextPathFromDocument(doc);
        if (newFoundNextPath) {
          currentNextPath = newFoundNextPath;
        } else {
          console.log('No more "next" links found in fetched content. End of posts.');
          noMorePosts = true;
          if (sentinel && observer) observer.unobserve(sentinel);
        }
      } else {
        console.log('No new posts found in fetched content wrapper. Assuming end of content.');
        noMorePosts = true;
        if (sentinel && observer) observer.unobserve(sentinel);
      }
    } catch (error) {
      console.error('Error loading more posts:', error);
      noMorePosts = true; 
      if (sentinel && observer) observer.unobserve(sentinel);
    } finally {
      if(spinner.parentNode) spinner.remove();
      isLoading = false;
      if (noMorePosts && sentinel && sentinel.parentNode) { 
          if(observer) observer.unobserve(sentinel); // Ensure observer is defined
          sentinel.remove();
          sentinel = null; // Nullify to prevent errors if somehow called again
      }
    }
  };

  const observerCallback = (entries) => {
    const entry = entries[0];
    if (entry.isIntersecting && !isLoading && !noMorePosts && currentNextPath) {
      loadMorePosts();
    }
  };
  
  observer = new IntersectionObserver(observerCallback, { 
    rootMargin: '0px 0px 200px 0px',
    threshold: 0.01 
  });

  sentinel = document.createElement('div');
  sentinel.id = 'infinite-scroll-sentinel';
  sentinel.style.height = '10px'; 
  // Insert sentinel after the post container.
  // If postContainer is the last child of its parent, this works.
  // If there were other elements after postContainer (like a footer within the same parent),
  // it should be postContainer.parentNode.insertBefore(sentinel, postContainer.nextSibling);
  // However, given pagination is hidden and likely was the next sibling, appending to parent is okay.
  postContainer.parentNode.appendChild(sentinel); 
  observer.observe(sentinel);

}); 