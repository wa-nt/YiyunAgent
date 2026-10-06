from app.resources import resource_path


HTML = resource_path("web/index.html").read_text(encoding="utf-8")


def _sidebar() -> str:
    return HTML.split('<aside class="sidebar">', 1)[1].split("</aside>", 1)[0]


def _settings_panel() -> str:
    return HTML.split('id="settings-panel"', 1)[1].split('id="tasks-mask"', 1)[0]


def test_sidebar_has_top_action_entries():
    sidebar = _sidebar()

    assert 'id="new-session"' in sidebar
    assert 'id="tasks-btn"' in sidebar
    assert 'id="skills-btn"' in sidebar
    assert "新对话" in sidebar
    assert "定时任务" in sidebar
    assert "技能" in sidebar


def test_sidebar_drops_ingest_docs_memory_gap_sections():
    sidebar = _sidebar()

    # 导入 / 文档 / 记忆 / 漏洞 不再是侧栏分区
    assert 'id="ingest-input"' not in sidebar
    assert 'id="doc-list"' not in sidebar
    assert 'id="memory-list"' not in sidebar
    assert 'id="gap-list"' not in sidebar
    assert "知识库文档" not in sidebar


def test_sidebar_has_no_colour_emoji():
    sidebar = _sidebar()

    for emoji in ("💬", "📥", "📁", "🧠", "⚠", "⏰"):
        assert emoji not in sidebar


def test_settings_panel_has_section_nav():
    panel = _settings_panel()

    for section in ("界面", "通用", "供应商", "技能", "插件", "用量统计", "关于"):
        assert section in panel


def test_settings_panel_hosts_memory_and_gap_lists():
    panel = _settings_panel()

    assert 'id="memory-list"' in panel
    assert 'id="gap-list"' in panel


def test_composer_plus_button_opens_file_picker_directly():
    # + 按钮不再跳导入框，直接触发文件选择
    assert 'id="file-input"' in HTML
    assert '$("file-input").click()' in HTML


def test_plugin_section_shows_empty_placeholder():
    panel = _settings_panel()

    assert "暂无插件" in panel


def test_appearance_section_has_real_controls():
    panel = _settings_panel()
    page = panel.split('data-page="appearance"', 1)[1].split('data-page="general"', 1)[0]

    # 主题 + 正文字号 + 对话宽度：都是能真实生效的界面设置
    assert 'id="set-dark"' in page
    assert 'id="set-font-size"' in page
    assert 'id="set-chat-width"' in page


def test_general_section_groups_startup_and_notifications():
    panel = _settings_panel()
    page = panel.split('data-page="general"', 1)[1].split('data-page="provider"', 1)[0]

    assert "启动与后台" in page
    assert 'id="set-autostart"' in page
    assert 'id="set-notify"' in page


def test_about_section_shows_version_and_license():
    panel = _settings_panel()
    page = panel.split('data-page="about"', 1)[1].split('class="set-actions"', 1)[0]

    assert 'id="about-version"' in page
    assert "许可证" in page
    assert "版权" in page


def test_sidebar_session_tools_have_new_search_filter():
    sidebar = _sidebar()

    assert 'id="sess-new"' in sidebar
    assert 'id="sess-search"' in sidebar
    assert 'id="sess-filter"' in sidebar
    assert 'id="sess-filter-pop"' in sidebar
    assert 'id="sess-sort"' in sidebar
    assert 'id="sess-hide-empty"' in sidebar


def test_usage_section_has_heatmap():
    panel = _settings_panel()
    page = panel.split('data-page="usage"', 1)[1].split('data-page="about"', 1)[0]

    assert 'id="heat-grid"' in page
    for metric in ("tokens_total", "calls", "cost"):
        assert f'data-metric="{metric}"' in page


def test_usage_section_has_stat_cards_and_request_log():
    panel = _settings_panel()
    page = panel.split('data-page="usage"', 1)[1].split('data-page="about"', 1)[0]

    # 顶部统计卡片 + 下方请求日志表（列只用 /api/traces 真有的字段）
    assert 'id="usage-cards"' in page
    assert 'id="trace-log"' in page
    for col in ("时间", "类型", "名称", "输入", "输出", "成本"):
        assert col in page


def test_scheduled_session_badge_uses_svg_clock_not_emoji():
    assert 'title="定时任务创建"' in HTML
    assert "定时任务创建\">⏰" not in HTML
