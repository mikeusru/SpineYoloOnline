// Configure Pusher instance
const pusher = new Pusher('309568955ad8ba7e672c', {
    cluster: 'us2',
    encrypted: true
});
const uID = Math.floor((Math.random()*100)+1);


    $(document).ready(function(){
      var dataTable = $("#dataTable").DataTable()
      var customerChannel = pusher.subscribe('customer');
      customerChannel.bind('add', function(data) {
      var date = new Date();
      dataTable.row.add([
          data.size,
          data.scale,
          data.count,
          data.coord_file
        ]).draw( false );
      });
    });