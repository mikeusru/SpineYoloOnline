var progressChannel = pusher.subscribe('progress' + uID);
progressChannel.bind('update', function(data) {
  //data = {message: "hello you", progress:42}
    var messageBox = $('#create-account-form-with-realtime').children('.messages');
    var progressBar = $('#realtime-progress-bar');

    messageBox.html(data.message);
    progressBar.width(data.progress+"%");
});
