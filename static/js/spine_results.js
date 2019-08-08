// Configure Pusher instance
const pusher = new Pusher('309568955ad8ba7e672c', {
    cluster: 'us2',
    encrypted: true
});


    $(document).ready(function(){
      var dataTable = $("#dataTable").DataTable()
      var customerChannel = pusher.subscribe('spine_results' + uID);
      customerChannel.bind('add', function(data) {
      var date = new Date();
      dataTable.row.add([
          data.size,
          data.scale,
          data.count,
          'Download CSV'.link(data.boxes_file)
        ]).draw( false );
      });
    });