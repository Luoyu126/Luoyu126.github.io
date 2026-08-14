require "cgi"
require "json"

module Jekyll
  class LegacyArticlesRedirectPage < PageWithoutAFile
    def initialize(site, legacy_path, target_path)
      directory = legacy_path.sub(%r{\A/}, "").sub(%r{/\z}, "")
      super(site, site.source, directory, "index.html")

      relative_target = "#{site.baseurl}#{target_path}"
      canonical_target = "#{site.url}#{relative_target}"
      escaped_relative_target = CGI.escapeHTML(relative_target)
      escaped_canonical_target = CGI.escapeHTML(canonical_target)

      self.data = { "layout" => nil, "sitemap" => false }
      self.content = <<~HTML
        <!doctype html>
        <html lang="en">
          <head>
            <meta charset="utf-8">
            <meta name="robots" content="noindex">
            <link rel="canonical" href="#{escaped_canonical_target}">
            <meta http-equiv="refresh" content="0; url=#{escaped_relative_target}">
            <script>window.location.replace(#{relative_target.to_json} + window.location.search + window.location.hash);</script>
            <title>Redirecting to Articles</title>
          </head>
          <body>
            <p>This page has moved to <a href="#{escaped_relative_target}">#{escaped_relative_target}</a>.</p>
          </body>
        </html>
      HTML
    end
  end

  class LegacyArticlesRedirectGenerator < Generator
    safe true
    priority :lowest

    def generate(site)
      redirects = { "/blog/" => "/articles/" }

      site.posts.docs.each do |post|
        next unless post.url.start_with?("/articles/")

        redirects[post.url.sub(%r{\A/articles/}, "/blog/")] = post.url
        year = post.date.strftime("%Y")
        redirects["/blog/#{year}/"] = "/articles/#{year}/"
      end

      site.tags.each_key do |tag|
        slug = Utils.slugify(tag)
        redirects["/blog/tag/#{slug}/"] = "/articles/tag/#{slug}/"
      end

      site.categories.each_key do |category|
        slug = Utils.slugify(category)
        redirects["/blog/category/#{slug}/"] = "/articles/category/#{slug}/"
      end

      redirects.each do |legacy_path, target_path|
        site.pages << LegacyArticlesRedirectPage.new(site, legacy_path, target_path)
      end
    end
  end
end
