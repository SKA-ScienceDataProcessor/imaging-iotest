

from __future__ import print_function
import datetime

import re
import sys

if len(sys.argv) < 2:
    print("Please provide SLURM output file!")
    exit(1)

rank_output = {}
def add_output(rank, out):
    if not rank in rank_output:
        rank_output[rank] = []
    rank_output[rank].append(out)

print("Reading %s..." % sys.argv[1])
p = re.compile("(\[[\d,]*\])(<\w*>:| )")
with open(sys.argv[1], "r") as f:
    for line in f:
        elems = p.split(line)
        add_output('', elems[0]);
        for i in range(len(elems)//3):
            add_output(elems[i*3+1], elems[i*3+3])


def natural_keys(text):
    # Nicked from https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    return [ int(c) if c.isdigit() else c
             for c in re.split(r'(\d+)', text) ]

show_ranks = "--show-ranks" in sys.argv
if "--dump" in sys.argv:
    for rank in sorted(rank_output.keys(), key=natural_keys):
        output = rank_output[rank]
        print("\n ** Rank %s **\n" % rank)
        for out in output:
            print(out, end='')

print("\nConfiguration\n-------------")

# Determine configuration
for out in rank_output.get('',[]):
    if out.startswith('Time: ') or out.startswith('Finish time:'):
        try:
            time = datetime.datetime.strptime(out[out.index(':')+2:-1], "%a %d %b %H:%M:%S %Z %Y")
        except:
            time = datetime.datetime.strptime(out[out.index(':')+2:-1], "%a %b %d %H:%M:%S %Z %Y")
        epoch_start = datetime.datetime(1970,1,1)
        epoch = int( (time - epoch_start).total_seconds() * 1000 )
        if out.startswith('Time'):
            start_time = time
            print("Start Epoch: {:d}".format(epoch))
        else:
            print("Finish Epoch: {:d}".format(epoch))
            print("Duration: {}s".format((time - start_time).total_seconds()))
            print()

tf_p = re.compile(".*--time=[\+\-\d:.e]+/([\d]+)/([\d]+).+--freq=[\+\-\d:.e]+/([\d]+)/([\d]+)")
fs_p = re.compile("^[\d\.]+@[^\s]+\s+([^\s]+)\s+([^\s]+)")
for out in rank_output.get('',[]):
    if out.startswith('OMP') or out.startswith('\(OMP') or \
       out.startswith('numtasks') or out.startswith('++ mpirun'):
        print(out, end='')
    if out.startswith('mpirun'):
        print("Command:", out, end='')
        match = tf_p.match(out)
        if match:
            t_count = int(match.group(1)); t_chunk = int(match.group(2))
            f_count = int(match.group(3)); f_chunk = int(match.group(4))
            print("Chunks: %g KiB" % (16 * t_chunk * f_chunk / 1024.))
    match = fs_p.match(out)
    if match:
        print("Storage: %s (%s used)" % (match.group(1), match.group(2)))

# Determine roles
producers = {}
producers_pid = {}
streamers = {}
streamers_pid = {}
writers = {}
writer_ids = {}
p = re.compile("(.*) role: (\w*) (\d*)")
p_w = re.compile("Writer (\d*):")
for rank, output in rank_output.items():
    for out in output:
        match = p.match(out)
        if match:
            pid = match.group(1)
            role = match.group(2)
            role_id = int(match.group(3))
            if role == 'Producer':
                producers[role_id] = output
                producers_pid[role_id] = pid
            elif role == 'Streamer':
                streamers[role_id] = output
                streamers_pid[role_id] = pid
            else:
                print("Unknown role of %s: %s", rank, role)
            continue
        match = p_w.match(out)
        if match:
            writer_id = int(match.group(1))
            if not writer_id in writers:
                writers[writer_id] = []
                writer_ids[writer_id] = writer_id
            writers[writer_id].append(out)

producer_ids = list(sorted(producers.keys()))
streamer_ids = list(sorted(streamers.keys()))
if show_ranks:
    print("%d Producers (%s)" % (len(producers), ", ".join(["%d at %s" % (pid, producers_pid[pid]) for pid in producer_ids])))
    print("%d Streamers (%s)" % (len(streamers), ", ".join(["%d at %s" % (sid, streamers_pid[sid]) for sid in streamer_ids])))
else:
    print("%d Producers, %d Streamers" % (len(producers), len(streamers)))
if len(writers) > 0:
    print("%d Writer Threads" % len(writers))

# Look for errors
got_errors = False
p = re.compile("^(HDF5-DIAG:|  #000:|  #.*error message|[\w:\d]* terminated with signal)")
for rank, output in sorted(rank_output.items(), key=lambda x: x[0]):
    got_rank_error = False
    for out in output:
        if p.match(out):
            if not got_errors:
                print("\nErrors\n------")
                got_errors = True
            if not got_rank_error:
                print(" ** Rank", rank, "**")
                got_rank_error = True
            print(out, end='')

# Extract timings
stream_create_time = {}
stream_stream_time = {}
stream_received = {}
stream_written = {}
stream_rewritten = {}
stream_receiver_wait = {}
stream_worker_wait = {}
stream_writer_wait = {}
stream_writer_read = {}
stream_writer_write = {}
stream_extracts = [
    ("Create", "s", stream_create_time, re.compile("^done in ([\d\.+-e]*)s")),
    ("Stream", "s", stream_stream_time, re.compile("^Streamed for ([\d\.+-e]*)s")),
    ("Received", " GB", stream_received, re.compile("^Received ([\d\.+-e]*) GB")),
    ("Receiver Wait", "s", stream_receiver_wait, re.compile("^Receiver: Wait: ([\d\.+-e]*)s")),
    ("\nWorker Wait", "s", stream_worker_wait, re.compile("^Worker: Wait: ([\d\.+-e]*)s")),
    ("Worker Degrid", "s", {}, re.compile("^Worker: .*Degrid: ([\d\.+-e]*)s")),
    ("Worker Idle", "s", {}, re.compile("^Worker: .*Idle: ([\d\.+-e]*)s")),
    ("Degrid rate", " Gflop/s", {}, re.compile("^Operations: degrid ([\d\.+-e]*)")),
    ("Accuracy RMSE", "", {}, re.compile("^Accuracy: RMSE ([\d\.+\-e]*)")),
    ("Accuracy worst", "", {}, re.compile("^Accuracy: .*worst ([\d\.+-e]*)")),
]
if len(writers) == 0:
    stream_extracts.extend([
        ("\nWritten", " GB", stream_written, re.compile("^Written ([\d\.+-e]*) GB")),
        ("Rewritten", " GB", stream_rewritten, re.compile("^Written .*rewritten ([\d\.+-e]*) GB")),
        ("Writer Wait", "s", stream_writer_wait, re.compile("^Writer: Wait: ([\d\.+-e]*)s")),
        ("Writer Read", "s", stream_writer_read, re.compile("^Writer: .*Read: ([\d\.+-e]*)s")),
        ("Writer Write", "s", stream_writer_write, re.compile("^Writer: .*Write: ([\d\.+-e]*)s")),
    ])
    writer_extracts = []
else:
    writer_extracts = [
        ("\nWritten", " GB", stream_written, re.compile("^Writer \d*: ([\d\.+-e]*) GB")),
        ("Rewritten", " GB", stream_rewritten, re.compile("^Writer \d*: .*rewritten ([\d\.+-e]*) GB")),
        ("Writer Wait", "s", stream_writer_wait, re.compile("^Writer \d*: Wait: ([\d\.+-e]*)s")),
        ("Writer Read", "s", stream_writer_read, re.compile("^Writer \d*: .*Read: ([\d\.+-e]*)s")),
        ("Writer Write", "s", stream_writer_write, re.compile("^Writer \d*: .*Write: ([\d\.+-e]*)s")),
    ]

producer_extracts = [
    ("PF1", "s", {}, re.compile("^PF1: ([\d\.+-e]*)")),
    ("FT1", "s", {}, re.compile("^PF1:.*FT1: ([\d\.+-e]*)")),
    ("ES1", "s", {}, re.compile("^PF1:.*ES1: ([\d\.+-e]*)")),
    ("PF2", "s", {}, re.compile("^PF2: ([\d\.+-e]*)")),
    ("FT2", "s", {}, re.compile("^PF2:.*FT2: ([\d\.+-e]*)")),
    ("ES2", "s", {}, re.compile("^PF2:.*ES2: ([\d\.+-e]*)")),
]

for producer_id, output in producers.items():
    for out in output:
        for _, _, dct, p in producer_extracts:
            m = p.match(out)
            if m: dct[producer_id] = float(m.group(1))
for writer_id, output in writers.items():
    for out in output:
        for _, _, dct, p in writer_extracts:
            m = p.match(out)
            if m: dct[writer_id] = float(m.group(1))
for streamer_id, output in streamers.items():
    for out in output:
        for _, _, dct, p in stream_extracts:
            m = p.match(out)
            if m: dct[streamer_id] = float(m.group(1))

for extracts, ids, ex_name in [(stream_extracts, streamer_ids, "Streamer"),
                               (writer_extracts, writer_ids, "Writer"),
                               (producer_extracts, producer_ids, "Producer")]:
  print("\n%s Stats\n-----" % ex_name)
  for name, unit, dct, _ in extracts:
    if len(dct) == 0:
        print(" !!! \"%s\" missing !!!" % name.replace('\n',''))
        continue
    print("%s: %g%s total (%s%g%s min, %g%s max, %g%s average)" % (
        name,
        sum(dct.values()), unit,
        "" if len(dct.values()) == len(ids) else "%d/%d missing - " % (
            len(ids) - len(dct.values()), len(ids)),
        min(dct.values()), unit,
        max(dct.values()), unit,
        sum(dct.values())/len(dct.values()), unit))
    if show_ranks:
        def format_val(sid):
            if sid in dct:
                return "[%s] %g" % (sid, dct[sid])
            else:
                return "[%s] ---" % sid
        print("  (%s)" % ", ".join([format_val(sid) for sid in ids]))
