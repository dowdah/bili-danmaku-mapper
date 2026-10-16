ObjC.import('AppKit');
function run(args) {
  var app = $.NSApplication.sharedApplication;
  app.setActivationPolicy($.NSApplicationActivationPolicyAccessory);
  app.activateIgnoringOtherApps(true);
  var panel = $.NSOpenPanel.openPanel;
  var directory = args[0] === 'directory';
  panel.canChooseDirectories = directory;
  panel.canChooseFiles = !directory;
  var danmaku = args[0] === 'danmaku';
  var subtitle = args[0] === 'subtitle';
  panel.allowsMultipleSelection = args[0] === 'files';
  panel.title = directory ? '选择未审查影片目录' : subtitle ? '选择外挂字幕（SRT / VTT / ASS / SSA）' : danmaku ? '选择已对齐的弹幕 JSON' : args[0] === 'playback_video' ? '选择成品影片' : '选择未审查影片';
  if (!directory) panel.allowedFileTypes = subtitle ? ['srt','vtt','ass','ssa'] : danmaku ? ['json'] : ['mp4','mkv','mov','avi','flv','webm','m4v'];
  if (panel.runModal !== $.NSModalResponseOK) return '[]';
  var paths = [];
  var urls = panel.URLs;
  for (var i = 0; i < urls.count; i++) paths.push(ObjC.unwrap(urls.objectAtIndex(i).path));
  return JSON.stringify(paths);
}
