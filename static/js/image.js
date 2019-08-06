
$(document).ready(function () {
  $('.navbar-sidenav [data-toggle="tooltip"]').tooltip({
    template: '<div class="tooltip navbar-sidenav-tooltip" role="tooltip" style="pointer-events: none;"><div class="arrow"></div><div class="tooltip-inner"></div></div>'
  })

  $('[data-toggle="tooltip"]').tooltip()

  var imageChannel = pusher.subscribe('image');
  imageChannel.bind('send', function(data) {
    var toAppend = document.createElement('a')
    document.getElementById('spine-image-box').appendChild(toAppend)
    toAppend.innerHTML ='<div class="media">'+
                    '<div class="media-body">'+
                      `<img src = "{{ url_for('static', filename=${data.image_link})}}" alt="Analyzed Image">`,
                    '</div>'+
                  '</div>'

  });
});