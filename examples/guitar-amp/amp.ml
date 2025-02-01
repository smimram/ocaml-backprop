(** Pre-emphasis for error computation. *)

let () =
  Printexc.record_backtrace true;
  let error fmt = Printf.ksprintf (fun s -> print_endline s; exit 1) fmt in
  let source = ref "" in
  let target = ref "" in
  let output = ref "output.wav" in
  let rate = ref 0.005 in
  let size = ref 20 in
  let play = ref false in
  let bulk = 1024 in
  Arg.parse [
    "-i", Arg.Set_string source, "Input file.";
    "-s", Arg.Set_string source, "Source file.";
    "-t", Arg.Set_string target, "Target file.";
    "-o", Arg.Set_string output, "Output file.";
    "-r", Arg.Set_float rate, "Learning rate.";
    "--rate", Arg.Set_float rate, "Learning rate.";
    "--size", Arg.Set_int size, "Size of the network.";
    "-p", Arg.Set play, "Play processed data on soundcard.";
    "--play", Arg.Set play, "Play processed data on soundcard.";
  ] (fun s -> source := s) "learn [options]";
  if !source = "" then error "Please specify an input file.";
  let source = WAV.openfile !source in
  let _channels = WAV.channels source in
  let samplerate = WAV.samplerate source in
  let _samples = WAV.samples source in
  if !target = "" then Printf.printf "Playing:\n" else Printf.printf "Learning:\n";
  Printf.printf "- rate: %f\n" !rate;
  if !target = "" then
    (
      print_endline "File processing mode."
    ) (* TODO: process file *)
    (*
    (
      let output = Output.create ~channels ~samplerate ~filename:!output ~soundcard:!play () in
      let json = Yojson.Basic.from_file !json in
      let net = Array.init channels (fun _ -> Net.of_json json) in
      try
        let i = ref 0 in
        while true do
          Printf.printf "\rProcessing: %2.00f%%%!" (100. *. float !i /. float samples);
          incr i;
          let x = WAV.sample_float source in
          let y = Array.init channels (fun c -> Net.process net.(c) x.(c)) in
          Output.sample output y
        done
      with End_of_file -> Printf.printf "\rDone!\n%!"
    )
       *)
  else
    (
      print_endline "Learning mode.";
      Random.self_init ();
      let target = WAV.openfile !target in
      let output = Output.create ~channels:1 ~samplerate ~filename:!output ~soundcard:!play () in
      let net = Net.net in
      let state = ref (Net.state ()) in
      try
        let i = ref 0 in
        ignore (Sys.signal Sys.sigint (Signal_handle (fun _ -> raise End_of_file)));
        (* let pec = preemph () in *)
        (* let pet = preemph () in *)
        while true do
          incr i;
          let x = WAV.samples_float source bulk |> Array.map (fun x -> x.(0)) in
          let y = WAV.samples_float target bulk |> Array.map (fun x -> x.(0)) in
          let s', y' = net !state y x in
          state := s';
          Output.samples output y';
          (* (\* Printf.printf "S: %.02f\tT: %.02f\tC: %.02f\n" x yt yc; *\) *)
          (* (\* let yc = pec yc in *\) *)
          (* (\* let yt = pet yt in *\) *)
          (* let d = yc -. yt in *)
          (* Net.descent net (!rate *. d); *)
          (* Printf.printf "\rProcessing: %2.00f%% (err: %7.04f)%!" (100. *. float !i /. float samples) d *)
        done;
      with
      | End_of_file -> ()
    )
