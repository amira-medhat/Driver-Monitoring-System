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

    //settings button click event
    $("#SettingsBtn").click(function () {
        eel.playClickSound();
        $("#Oval").attr("hidden", false);
        $("#SiriWave").attr("hidden", true);
    });
    
    // //chat button click event
    // $("#PlayBtn").click(function () {
    //     eel.playClickSound();
    //     $("#Oval").attr("hidden", false);
    //     $("#SiriWave").attr("hidden", true);
    //     isMonitoring = true;
    //     eel.speak("Assistant is now in monitoring mode...")
    //     eel.start_monitoring()();
    // });

    // //settings button click event
    // $("#ExitBtn").click(function () {
    //     eel.playClickSound();
    //     $("#Oval").attr("hidden", false);
    //     $("#SiriWave").attr("hidden", true);
    //     eel.speak("Assistant is exitting from mointoring mode...")
    // });

});