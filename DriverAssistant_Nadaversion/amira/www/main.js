$(document).ready(function () {

    let isMonitoring = false;  //Global flag to check if the assistant is in monitoring mode

    $('.text').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "fadeIn",
        },
        out: {
            effect: "fadeOutUp",
        },

    });

    //siriwave configuration
    var siriWave = new SiriWave({
        container: document.getElementById("siri-container"),
        width: 800,
        height: 200,
        style: "ios9",
        color: "#fff",
        speed: 0.2,
        amplitude: 1,
        autostart: true
      });

    //siri message animation
    $('.siri-message').textillate({
        loop: true,
        sync: true,
        in: {
            effect: "fadeInUp",
            sync: true,
        },
        out: {
            effect: "fadeOutUp",
            sync: true,
        },

    });

    //mic button click event
    $("#MicBtn").click(function () {
        eel.playClickSound();
        $("#Oval").attr("hidden", true);
        $("#SiriWave").attr("hidden", false);
        
        // Tell Python that the mic button was clicked
        eel.set_mic_pressed(); 
    });

    
    // Toggle Settings Window on Settings Button Click
    $("#SettingsBtn").click(function () {
        $("#SettingsWindow").fadeToggle(); // Show/hide window
    });

    // Close Settings Window
    $("#CloseSettings").click(function () {
        $("#SettingsWindow").fadeOut();
    });

    //Gps button click event
    $("#GpsBtn").click(function () {
        eel.playClickSound();
        $("#Oval").attr("hidden",false);
        $("#SiriWave").attr("hidden", true);
        eel.OpenGps("gps");
        //$("#MapScreen").attr("hidden", false);
    });

    //NEW
    // Update button UI based on backend flag
    function updateMonitorButtons() {
        eel.get_monitor_mode()(function (state) {
            if (state === "on") {
                $("#MonitorOnBtn").addClass("selected-option");
                $("#MonitorOffBtn").removeClass("selected-option");
            } else {
                $("#MonitorOffBtn").addClass("selected-option");
                $("#MonitorOnBtn").removeClass("selected-option");
            }
        });
    }
    // Call this on load
    $(document).ready(function () {
        updateMonitorButtons();
        // Start polling every 2 seconds
        setInterval(() => {
            updateMonitorButtons();
        }, 2000);

        $("#MonitorOnBtn").click(function () {
            $("#Oval").attr("hidden",false);
            $("#SiriWave").attr("hidden", true);
            eel.Set_jason_flag(); 
            updateMonitorButtons();
        });

        $("#MonitorOffBtn").click(function () {
            $("#Oval").attr("hidden",false);
            $("#SiriWave").attr("hidden", true);
            eel.Clear_jason_flag();
            updateMonitorButtons();
        });
    });


});