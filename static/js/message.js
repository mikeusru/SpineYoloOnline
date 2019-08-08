
$(document).ready(function () {
  $('.navbar-sidenav [data-toggle="tooltip"]').tooltip({
    template: '<div class="tooltip navbar-sidenav-tooltip" role="tooltip" style="pointer-events: none;"><div class="arrow"></div><div class="tooltip-inner"></div></div>'
  });

  $('[data-toggle="tooltip"]').tooltip();

  var messageChannel = pusher.subscribe('message' + uID);
  messageChannel.bind('send', function(data) {
    var date = new Date();
    var toAppend = document.createElement('a');
    toAppend.classList.add('list-group-item', 'list-group-item-action');
    toAppend.href = '#';
    document.getElementById('message-box').appendChild(toAppend);
    toAppend.innerHTML ='<div class="media">'+
                    '<img class="d-flex mr-3 rounded-circle" src="http://placehold.it/45x45" alt="">'+
                    '<div class="media-body">'+
                      `<strong>${data.name}</strong> posted a new message `+
                      `<em>${data.message}</em>.`+
                      `<div class="text-muted smaller">Today at ${date.getHours()} : ${date.getMinutes()}</div>`+
                    '</div>'+
                  '</div>';

  });
});