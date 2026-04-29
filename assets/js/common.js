$(document).ready(function () {
  // add toggle functionality to abstract, award and bibtex buttons
  $("a.abstract").click(function () {
    $(this).parent().parent().find(".abstract.hidden").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.award").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden.open").toggleClass("open");
  });
  $("a.bibtex").click(function () {
    $(this).parent().parent().find(".abstract.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".award.hidden.open").toggleClass("open");
    $(this).parent().parent().find(".bibtex.hidden").toggleClass("open");
  });
  $("a").removeClass("waves-effect waves-light");

  // bootstrap-toc
  if ($("#toc-sidebar").length) {
    // remove related publications years from the TOC
    $(".publications h2").each(function () {
      $(this).attr("data-toc-skip", "");
    });
    var navSelector = "#toc-sidebar";
    var $myNav = $(navSelector);
    Toc.init($myNav);
    $("body").scrollspy({
      target: navSelector,
      offset: 100,
    });
  }

  const tocToggleButton = document.getElementById("toc-sidebar-toggle");
  const tocFloatingToggle = document.getElementById("toc-floating-toggle");
  const tocSidebar = document.getElementById("toc-sidebar");
  if ((tocToggleButton || tocFloatingToggle) && tocSidebar) {
    const tocLayout =
      tocToggleButton?.closest(".toc-layout") ||
      tocFloatingToggle?.closest(".toc-layout") ||
      document.querySelector(".toc-layout");
    const tocMainColumn = tocLayout?.querySelector(".toc-main-column");
    const tocSidebarColumn = tocLayout?.querySelector(".toc-sidebar-column");

    const setTocCollapsed = (isCollapsed) => {
      tocSidebar.classList.toggle("is-collapsed", isCollapsed);

      if (tocLayout && tocMainColumn && tocSidebarColumn) {
        tocLayout.classList.toggle("toc-collapsed", isCollapsed);
        tocMainColumn.classList.toggle("col-sm-9", !isCollapsed);
        tocMainColumn.classList.toggle("col-sm-12", isCollapsed);
        tocSidebarColumn.classList.toggle("col-sm-3", !isCollapsed);
        tocSidebarColumn.classList.toggle("d-none", isCollapsed);
      }

      if (tocToggleButton) {
        tocToggleButton.setAttribute("aria-expanded", (!isCollapsed).toString());
        tocToggleButton.textContent = "Hide TOC";
      }

      if (tocFloatingToggle) {
        tocFloatingToggle.classList.toggle("show", isCollapsed);
      }
    };

    if (tocToggleButton) {
      tocToggleButton.addEventListener("click", () => setTocCollapsed(true));
    }

    if (tocFloatingToggle) {
      tocFloatingToggle.addEventListener("click", () => setTocCollapsed(false));
    }
  }

  // add css to jupyter notebooks
  const cssLink = document.createElement("link");
  cssLink.href = "../css/jupyter.css";
  cssLink.rel = "stylesheet";
  cssLink.type = "text/css";

  let jupyterTheme = determineComputedTheme();

  $(".jupyter-notebook-iframe-container iframe").each(function () {
    $(this).contents().find("head").append(cssLink);

    if (jupyterTheme == "dark") {
      $(this).bind("load", function () {
        $(this).contents().find("body").attr({
          "data-jp-theme-light": "false",
          "data-jp-theme-name": "JupyterLab Dark",
        });
      });
    }
  });

  // trigger popovers
  $('[data-toggle="popover"]').popover({
    trigger: "hover",
  });
});
