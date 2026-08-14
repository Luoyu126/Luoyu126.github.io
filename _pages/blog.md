---
layout: default
permalink: /blog/
title: articles
nav: true
nav_order: 1
pagination:
  enabled: false
---

<div class="post">

{% assign blog_name_size = site.blog_name | size %}
{% assign blog_description_size = site.blog_description | size %}

{% if blog_name_size > 0 or blog_description_size > 0 %}

  <div class="header-bar">
    <h1>{{ site.blog_name }}</h1>
    {% if blog_description_size > 0 %}
      <h2>{{ site.blog_description }}</h2>
    {% endif %}
  </div>
  {% endif %}

{% if site.display_tags and site.display_tags.size > 0 or site.display_categories and site.display_categories.size > 0 %}

  <div class="tag-category-list">
    <ul class="p-0 m-0">
      {% for tag in site.display_tags %}
        <li>
          <i class="fa-solid fa-hashtag fa-sm"></i> <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">{{ tag }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
      {% if site.display_categories.size > 0 and site.display_tags.size > 0 %}
        <p>&bull;</p>
      {% endif %}
      {% for category in site.display_categories %}
        <li>
          <i class="fa-solid fa-tag fa-sm"></i> <a href="{{ category | slugify | prepend: '/blog/category/' | relative_url }}">{{ category }}</a>
        </li>
        {% unless forloop.last %}
          <p>&bull;</p>
        {% endunless %}
      {% endfor %}
    </ul>
  </div>
  {% endif %}

{% assign featured_posts = site.posts | where: "featured", "true" %}
{% if featured_posts.size > 0 %}
<br>

<div class="container featured-posts">
{% assign is_even = featured_posts.size | modulo: 2 %}
<div class="row row-cols-{% if featured_posts.size <= 2 or is_even == 0 %}2{% else %}3{% endif %}">
{% for post in featured_posts %}
<div class="col mb-4">
<a href="{{ post.url | relative_url }}">
<div class="card hoverable">
<div class="row g-0">
<div class="col-md-12">
<div class="card-body">
<div class="float-right">
<i class="fa-solid fa-thumbtack fa-xs"></i>
</div>
<h3 class="card-title text-lowercase">{{ post.title }}</h3>
<p class="card-text">{{ post.description }}</p>

                    {% assign year = post.date | date: "%Y" %}

                    <p class="post-meta">
                      <a href="{{ year | prepend: '/blog/' | relative_url }}">
                        <i class="fa-solid fa-calendar fa-sm"></i> {{ year }} </a>
                    </p>
                  </div>
                </div>
              </div>
            </div>
          </a>
        </div>
      {% endfor %}
      </div>
    </div>
    <hr>

{% endif %}

  <div class="blog-sort-controls">
    <label for="blog-sort-select">Sort by</label>
    <select id="blog-sort-select" aria-label="Sort blog posts">
      <option value="time" selected>Time</option>
      <option value="hot">Hot (Likes)</option>
    </select>
  </div>

  <ul class="post-list">

    {% assign postlist = site.posts %}

    {% for post in postlist %}
    {% assign year = post.date | date: "%Y" %}
    {% assign tags = post.tags | join: "" %}
    {% assign stars = post.stars | default: 0 %}
    {% assign article_type = post.article_type | default: "Blog" %}
    {% assign like_key_source = post.id | default: post.url %}
    {% assign like_key = like_key_source | replace: '/', '--' | replace: '.', '-' | replace: ':', '-' %}

    <li class="post-list-item" data-date="{{ post.date | date: '%s' }}" data-stars="{{ stars }}" data-like-key="{{ like_key }}">

{% if post.thumbnail %}

<div class="row">
          <div class="col-sm-9">
{% endif %}
        <div class="article-type-badge">{{ article_type }}</div>
        <h3>
        {% if post.redirect == blank %}
          <a class="post-title" href="{{ post.url | relative_url }}">{{ post.title }}</a>
        {% elsif post.redirect contains '://' %}
          <a class="post-title" href="{{ post.redirect }}" target="_blank">{{ post.title }}</a>
          <svg width="2rem" height="2rem" viewBox="0 0 40 40" xmlns="http://www.w3.org/2000/svg">
            <path d="M17 13.5v6H5v-12h6m3-3h6v6m0-6-9 9" class="icon_svg-stroke" stroke="#999" stroke-width="1.5" fill="none" fill-rule="evenodd" stroke-linecap="round" stroke-linejoin="round"></path>
          </svg>
        {% else %}
          <a class="post-title" href="{{ post.redirect | relative_url }}">{{ post.title }}</a>
        {% endif %}
      </h3>
      <p>{{ post.description }}</p>
      <p class="post-meta">
        {{ post.date | date: '%B %d, %Y' }}
        &nbsp; &middot; &nbsp;
        <i class="fa-regular fa-heart fa-sm"></i> <span class="post-like-count">{{ stars }}</span>
        {% if post.external_source %}
        &nbsp; &middot; &nbsp; {{ post.external_source }}
        {% endif %}
      </p>
      <p class="post-tags">
        <a href="{{ year | prepend: '/blog/' | relative_url }}">
          <i class="fa-solid fa-calendar fa-sm"></i> {{ year }} </a>

          {% if tags != "" %}
          &nbsp; &middot; &nbsp;
            {% for tag in post.tags %}
            <a href="{{ tag | slugify | prepend: '/blog/tag/' | relative_url }}">
              <i class="fa-solid fa-hashtag fa-sm"></i> {{ tag }}</a>
              {% unless forloop.last %}
                &nbsp;
              {% endunless %}
              {% endfor %}
          {% endif %}
    </p>

{% if post.thumbnail %}

</div>

  <div class="col-sm-3">
    <img class="card-img" src="{{ post.thumbnail | relative_url }}" style="object-fit: cover; height: 90%" alt="image">
  </div>
</div>
{% endif %}
    </li>

    {% endfor %}

  </ul>

<script>
  (() => {
    const sortSelect = document.getElementById("blog-sort-select");
    const postList = document.querySelector(".post-list");
    if (!sortSelect || !postList) return;
    const SUPABASE_URL = "{{ site.supabase.url | default: '' }}";
    const SUPABASE_ANON_KEY = "{{ site.supabase.anon_key | default: '' }}";
    const NORMALIZED_SUPABASE_URL = SUPABASE_URL.replace(/\/rest\/v1\/?$/, "").replace(/\/$/, "");
    const CACHE_KEY = "blog-like-cache-v1";
    const FETCH_TIMEOUT_MS = 2000;
    const hasSupabaseConfig = Boolean(NORMALIZED_SUPABASE_URL && SUPABASE_ANON_KEY);
    const postItems = Array.from(postList.querySelectorAll(".post-list-item"));

    const sortPosts = (sortBy) => {
      postItems.sort((a, b) => {
        const dateDiff = Number(b.dataset.date) - Number(a.dataset.date);
        if (sortBy === "time") return dateDiff;
        const starDiff = Number(b.dataset.stars) - Number(a.dataset.stars);
        return starDiff === 0 ? dateDiff : starDiff;
      });
      postItems.forEach((post) => postList.appendChild(post));
    };

    const readCache = () => {
      try {
        const parsed = JSON.parse(localStorage.getItem(CACHE_KEY) || "{}");
        return parsed && typeof parsed.counts === "object" ? parsed.counts : {};
      } catch (error) {
        return {};
      }
    };

    const writeCache = (counts) => {
      localStorage.setItem(CACHE_KEY, JSON.stringify({ updatedAt: Date.now(), counts }));
    };

    const applyCounts = (counts) => {
      postItems.forEach((post) => {
        const slug = post.dataset.likeKey;
        const fallbackValue = Number(post.dataset.stars) || 0;
        const likeValue = Number(counts[slug]);
        const resolved = Number.isFinite(likeValue) ? likeValue : fallbackValue;
        post.dataset.stars = String(resolved);
        const countEl = post.querySelector(".post-like-count");
        if (countEl) countEl.textContent = String(resolved);
      });
    };

    const withTimeout = (promise, timeoutMs) => {
      return Promise.race([
        promise,
        new Promise((_, reject) => {
          setTimeout(() => reject(new Error("timeout")), timeoutMs);
        }),
      ]);
    };

    const loadSupabaseClient = async () => {
      const supabaseModule = await import("https://cdn.jsdelivr.net/npm/@supabase/supabase-js/+esm");
      return supabaseModule.createClient(NORMALIZED_SUPABASE_URL, SUPABASE_ANON_KEY, {
        auth: { persistSession: false },
      });
    };

    const hydrateLikes = async () => {
      const cachedCounts = readCache();
      applyCounts(cachedCounts);
      sortPosts(sortSelect.value);

      if (!hasSupabaseConfig) return;

      const slugs = postItems.map((post) => post.dataset.likeKey).filter(Boolean);
      if (!slugs.length) return;

      try {
        const supabase = await loadSupabaseClient();
        const query = supabase.from("post_stats").select("slug, likes").in("slug", slugs);
        const { data, error } = await withTimeout(query, FETCH_TIMEOUT_MS);
        if (error) throw error;

        const mergedCounts = { ...cachedCounts };
        slugs.forEach((slug) => {
          const fallback = Number(mergedCounts[slug]) || 0;
          mergedCounts[slug] = fallback;
        });
        (data || []).forEach((row) => {
          mergedCounts[row.slug] = Number(row.likes) || 0;
        });

        writeCache(mergedCounts);
        applyCounts(mergedCounts);
        sortPosts(sortSelect.value);
      } catch (error) {
        // 2s timeout or fetch failure: keep cached render, no blocking.
      }
    };

    sortSelect.addEventListener("change", (event) => {
      sortPosts(event.target.value);
    });

    hydrateLikes();
  })();
</script>

</div>
