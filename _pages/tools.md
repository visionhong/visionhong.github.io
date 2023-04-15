---
layout: archive
permalink: tools
title: "Tools"
types: posts

author_profile: true
sidebar:
  nav: "sidebar-category"
---

{% assign posts = site.categories['tools']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}