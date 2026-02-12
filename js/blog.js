/*
 * Blog rendering engine
 * - Fetches post index from posts/index.json
 * - Renders markdown with marked.js + KaTeX for math
 * - Hash-based routing: blog.html = list, blog.html#slug = post
 */

(function () {
  'use strict';

  var posts = [];

  // ---- Math extraction ----
  // Extract $$...$$ and $...$ blocks before marked processes them,
  // to prevent marked from mangling LaTeX syntax.

  function extractMath(text) {
    var blocks = [];
    // Display math: $$...$$
    text = text.replace(/\$\$([\s\S]*?)\$\$/g, function (_, math) {
      var id = blocks.length;
      blocks.push({ math: math.trim(), display: true });
      return '\n\n%%MATH_' + id + '%%\n\n';
    });
    // Inline math: $...$  (not greedy, single line)
    text = text.replace(/\$([^\$\n]+?)\$/g, function (_, math) {
      var id = blocks.length;
      blocks.push({ math: math.trim(), display: false });
      return '%%MATH_' + id + '%%';
    });
    return { text: text, blocks: blocks };
  }

  function restoreMath(html, blocks) {
    blocks.forEach(function (block, id) {
      var rendered;
      try {
        rendered = katex.renderToString(block.math, {
          displayMode: block.display,
          throwOnError: false,
          trust: true
        });
      } catch (e) {
        rendered = '<span class="math-error" title="' +
          e.message.replace(/"/g, '&quot;') + '">' +
          block.math.replace(/</g, '&lt;') + '</span>';
      }
      var placeholder = '%%MATH_' + id + '%%';
      // marked may wrap display math placeholder in <p> tags
      html = html.replace('<p>' + placeholder + '</p>', '<div class="math-display">' + rendered + '</div>');
      html = html.replace(placeholder, block.display
        ? '<div class="math-display">' + rendered + '</div>'
        : '<span class="math-inline">' + rendered + '</span>');
    });
    return html;
  }

  // ---- Markdown rendering ----

  function renderMarkdown(text) {
    var extracted = extractMath(text);
    var html = marked.parse(extracted.text);
    html = restoreMath(html, extracted.blocks);
    return html;
  }

  // ---- Post index ----

  function loadIndex() {
    return fetch('posts/index.json')
      .then(function (r) {
        if (!r.ok) throw new Error('HTTP ' + r.status);
        return r.json();
      })
      .then(function (data) { posts = data; return data; })
      .catch(function (err) {
        var container = document.getElementById('blog-list-container');
        if (window.location.protocol === 'file:') {
          container.innerHTML =
            '<p style="color:var(--text-secondary)">Cannot load posts over <code>file://</code>. ' +
            'Run a local server:<br><code>python3 -m http.server</code><br>' +
            'then open <a href="http://localhost:8000/blog.html">http://localhost:8000/blog.html</a></p>';
        } else {
          container.innerHTML = '<p>Failed to load posts: ' + err.message + '</p>';
        }
        return [];
      });
  }

  // ---- Render post list ----

  function renderList() {
    var container = document.getElementById('blog-list-container');
    if (!posts.length) {
      container.innerHTML = '<p>No posts found.</p>';
      return;
    }
    var html = '';
    posts.forEach(function (post) {
      html += '<article class="blog-entry">' +
        '<h2 class="blog-entry-title"><a href="blog.html#' + post.slug + '">' +
        escapeHtml(post.title) + '</a></h2>' +
        '<div class="blog-entry-meta">' +
        '<span class="blog-entry-date">' + escapeHtml(post.date) + '</span>' +
        '<span class="blog-entry-tag">' + escapeHtml(post.tag) + '</span>' +
        '</div>' +
        '<p class="blog-entry-summary">' + escapeHtml(post.summary) + '</p>' +
        '</article>';
    });
    container.innerHTML = html;
  }

  // ---- Render single post ----

  function renderPost(slug) {
    var post = posts.find(function (p) { return p.slug === slug; });
    if (!post) {
      showList();
      return;
    }

    // Update header
    document.getElementById('blog-title').textContent = post.title;
    document.getElementById('blog-subtitle').textContent = post.date + ' · ' + post.tag;
    var backLink = document.getElementById('back-link');
    backLink.href = 'blog.html';
    backLink.innerHTML = '&larr; All Posts';

    // Show post view, hide list
    document.getElementById('blog-list').style.display = 'none';
    document.getElementById('blog-post').style.display = '';

    var contentEl = document.getElementById('post-content');
    contentEl.innerHTML = '<p style="color: var(--text-light);">Loading...</p>';

    fetch(post.file)
      .then(function (r) {
        if (!r.ok) throw new Error('Failed to load post');
        return r.text();
      })
      .then(function (md) {
        contentEl.innerHTML = renderMarkdown(md);
        // Scroll to top
        window.scrollTo(0, 0);
      })
      .catch(function () {
        contentEl.innerHTML = '<p>Failed to load post.</p>';
      });
  }

  // ---- Show list view ----

  function showList() {
    document.getElementById('blog-title').textContent = 'Blog';
    document.getElementById('blog-subtitle').textContent = 'Notes on research, tutorials, and reading.';
    var backLink = document.getElementById('back-link');
    backLink.href = 'index.html';
    backLink.innerHTML = '&larr; Home';

    document.getElementById('blog-list').style.display = '';
    document.getElementById('blog-post').style.display = 'none';
  }

  // ---- Router ----

  function route() {
    var hash = window.location.hash.slice(1);
    if (hash && hash !== '') {
      renderPost(hash);
    } else {
      showList();
    }
  }

  // ---- Helpers ----

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
  }

  // ---- Init ----

  function waitForLibs(cb) {
    if (typeof marked !== 'undefined' && typeof katex !== 'undefined') {
      cb();
    } else {
      setTimeout(function () { waitForLibs(cb); }, 50);
    }
  }

  document.addEventListener('DOMContentLoaded', function () {
    waitForLibs(function () {
      loadIndex().then(function () {
        renderList();
        route();
      });
    });
  });

  window.addEventListener('hashchange', route);
})();
