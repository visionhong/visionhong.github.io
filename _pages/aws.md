---
layout: archive
permalink: aws
title: "AWS"
types: posts

author_profile: true
sidebar:
  nav: "sidebar-category"
---

{% assign posts = site.categories['aws']%}
{% for post in posts %}
  {% include archive-single.html type=page.entries_layout %}
{% endfor %}