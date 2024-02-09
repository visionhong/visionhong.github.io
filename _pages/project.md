---
# layout: archive
layout: collection
entries_layout: grid
permalink: project
title: "프로젝트"
types: posts

author_profile: true
sidebar:
  nav: "sidebar-category"
---

{% assign posts = site.categories['project']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}