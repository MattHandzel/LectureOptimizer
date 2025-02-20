# This file will take in a transcript and improve it by using an LLM to do the following:
# Remove text that is redundant or not useful
# Ex. "or even microservices which is another buzzord that I did not define here" should be replaced with "or even microservices".
# Improving the grammar and english ability of the speaker
# Ex.
# right
# it's a very
# open area
# and there
# is a very
# there's a scarcity
# of education
# resources
# in that area
#  -Lectures are filled with rambling and non-concise passages.
# Ex: This "So, today we are discussing cloud computing, which is this new buzzword that has been around for a few years. And so, we are going to try to unearth what cloud computing is all about. We will try to kind of sort of define the term cloud computing." Should be "Today, we're discussing cloud computing, a recent buzzword. We'll unearth what it's all about and attempt to define the term."

import csv
import argparse
from utils import *
from utils import query_ollama

from sentence_transformers import SentenceTransformer

from sklearn.metrics.pairwise import cosine_similarity


global model
model = None
MODEL_NAME = "llama3.2:3b"


"""
PHI OUTPUT

```
0: It's not a very large-scale data center, but it's a medium-scale size. The racks got put togeth
er very quickly. You saw how they brought in those racks all in one block.
2-4:
5: And now they're putting in all the wiring, which is for both the power as well as for the Ether
net. That's going on the top. Plus also, as we mentioned last time, cooling, right?
6-7:
8: Cooling is a major part of the volume of a data center. Those big cables, those big pipes that 
you see here, are the pipes carrying the air, hot air and cold air.
9:
10: Lights are important. If you want to go in and check what your machines are doing, racks are d
oing, lights are important.
11:
12: You can see that the machines are already there, but the work is not done.
13: The video is only about halfway through.
14: And so there's a lot of other parts that need to be put in for the data center to be operation
al. So they are making sure that all the cabling is in place.
15-16:
17: If you were to put together a data center like any of Google's or Facebook's, those are very l
arge-scale and have hundreds of thousands of computers or machines in them, then that might take m
uch longer than this. And of course, it might use many more workers than you see in this video.
18:
19: Almost done here.
"""


def parse_ollama_response(response):
    if "```\n" in response:
        response = response.split("```\n")[1]
    if "\n```" in response:
        response = response.split("\n```")[0]
    response = response.replace("SEGMENT:", "").strip()
    response = "".join(response.split("\n"))
    response = response.replace("```", "").strip()
    response = (
        response.replace('""', "<QUOTE>").replace('"', "").replace("<QUOTE>", '"')
    )
    response = response.replace("**", "").replace("__", "")

    return response


def parse_ollama_response_v2(response):

    if "```\n" in response:
        response = response.split("```\n")[1]
    if "\n```" in response:
        response = response.split("\n```")[0]
    response = response.replace("SEGMENT:", "").strip()
    response = response.replace("```", "").strip()
    response = (
        response.replace('""', "<QUOTE>").replace('"', "").replace("<QUOTE>", '"')
    )
    response = response.replace("**", "").replace("__", "")

    new_response = []
    for line in response:
        line = line.strip()
        line_numbers, text = line.split()[0]
        print(line_numbers)
        print(text)
        if "-" in line_numbers:
            start, end = line_numbers.split("-")
        else:
            start = end = line_numbers
        new_response.append((start, end, text))

    return new_response


def improve_transcript_with_llm_v2(segments):
    prompt = """Revise the lecture transcript adhering strictly to these rules:

1. **Edits MUST:**
   - Fix grammar/sentence structure WITHOUT changing original words
   - Keep ALL technical terms & phrases 
   - Remove filler words/phrases 
   - Merge lines ONLY when necessary for flow, using original wording
   - Preserve every number from original transcript - include blank lines for removed segments

2. **Format REQUIREMENTS:**
   - Input line numbers MUST be preserved in order (1-20)
   - Merged lines show combined numbers (1-2)
   - Output MUST have EXACTLY 20 lines matching input numbering

3. **Output FORBIDDEN from:**
   - Paraphrasing technical descriptions
   - Changing sentence order
   - Omitting numbers from original sequence

Respond ONLY with revised transcript between ```, NO commentary.
    """

    # process the segments in chunks of n
    CHUNK_SIZE = 20
    STRIDE = CHUNK_SIZE // 2

    improved_segments = []
    for i in range(0, len(segments), STRIDE):
        end = min(i + CHUNK_SIZE, len(segments))
        text = "\n".join(
            [
                str(i + 1) + ":\t" + seg["text"].strip()
                for i, seg in enumerate(segments[i:end])
            ]
        )
        print("Prompt: ", prompt)
        print("TRANSCRIPT:\n```\n" + text + "\n```")
        response = query_ollama(
            "TRANSCRIPT:\n```\n" + text + "\n```",
            prompt,
            model_name=MODEL_NAME,
        )
        print(response)
        # extract_transcript_segments = parse_ollama_response_v2(response)
        # for start_index, end_index, text in extract_transcript_segments:
        #     improved_segments.append((start_index, end_index, text))

    # After responses are extracted, combine the segments back together and figure out what to do with the segments
    output_segments = []
    for start_index, end_index, text in improved_segments:
        raise "bro this is not done yet"
        start_index = int(start_index)
        end_index = int(end_index)
        text = text.strip()
        output_segments.append(
            {
                "start": segments[start_index]["start"],
                "end": segments[end_index]["end"],
                "text": text,
            }
        )

    return better_segments


def improve_transcript_with_llm(segments):
    prompt = """You are a lecture optimization assistant. You have been given a segment of a transcript for a lecture. Your task is to improve the transcript. Process transcript segments following these rules:
    1. Remove filler words (you know, okay, right), off-topic content, and redundant phrases
    2. Improve grammar and sentence structure
    3. Never add new information not in original
    4. Reword sentences so they are more concise
    5. Use the same word choices or phrases as the original text
    6. Output your response in the same format as the input.
    7. Only respond with the text of the segment, do not add any additional text.
    
Be wary of missing context in the transcript.
    """
    print("Prompt: ", prompt)
    better_segments = []
    for seg in segments:
        seg["text"] = seg["text"].strip()
        response = query_ollama(
            "TRANSCRIPT SEGMENT:\n```" + '"' + seg["text"] + '"' + "\n```",
            prompt,
            model_name=MODEL_NAME,
        )
        if "<remove>" in response:
            seg["text"] = ""
        seg["text"] = parse_ollama_response(response)
        print(seg["text"])
        better_segments.append(seg)
    return better_segments


def get_coherence(prev_segment, current_segment):
    global model
    if model is None:
        model = SentenceTransformer("all-mpnet-base-v2")
    emb_prev = model.encode(prev_segment)
    emb_current = model.encode(current_segment)
    return cosine_similarity([emb_prev], [emb_current])[0][0]


def merge_on_punctuation(prev_segment, current_segment):
    return (
        not prev_segment.strip().endswith((".", "!", "?"))
        and len(prev_segment) + len(current_segment) < 500
    )


def merge_on_semantics(prev_segment, current_segment):
    coherence = get_coherence(prev_segment, current_segment)
    # print(
    #     "Text {} and {} coherence: {}".format(prev_segment, current_segment, coherence)
    # )
    return (len(prev_segment) + len(current_segment)) < 100 and coherence > 0.80


import re


def split_on_periods(segments):
    output_segments = []
    for seg in segments:
        start, end, text = seg["start"], seg["end"], seg["text"]
        # Split into sentences, preserving original character count including trailing whitespace
        sentences_with_whitespace = text.split(". ")
        sentences_info = []
        for i, s in enumerate(sentences_with_whitespace):
            if i < len(sentences_with_whitespace) - 1:
                s += "."

            stripped = s.strip()
            if stripped:
                sentences_info.append(
                    (stripped, len(s))
                )  # Use original length for calculation

        if not sentences_info:
            output_segments.append({"start": start, "end": end, "text": text})
            continue

        total_chars = sum(length for _, length in sentences_info)
        total_duration = end - start
        avg_time_per_char = total_duration / total_chars if total_chars else 0

        current_time = start
        for i, (sentence, length) in enumerate(sentences_info):
            if i == len(sentences_info) - 1:
                # Ensure last sentence ends at the original end time
                sentence_end = end
            else:
                duration = length * avg_time_per_char
                sentence_end = current_time + int(round(duration))
                sentence_end = min(sentence_end, end)  # Prevent exceeding original end

            # Ensure no zero or negative duration
            if sentence_end <= current_time:
                sentence_end = current_time + 1

            output_segments.append(
                {
                    "start": current_time,
                    "end": sentence_end,
                    "text": sentence,
                }
            )
            current_time = sentence_end

    return output_segments


def merge_transcript(segments, should_merge_function):
    merged = []
    current = None
    for seg in segments:
        if current is None:
            current = seg
        else:
            # Check if current segment ends with sentence-ending punctuation
            if should_merge_function(current["text"], seg["text"]):
                # Merge with next segment
                current["text"] += " " + seg["text"]
                current["end"] = seg["end"]
            else:

                merged.append(current)
                current = seg
    if current is not None:
        merged.append(current)
    return merged


parser = argparse.ArgumentParser(description="Improve a transcript.")
parser.add_argument("input_transcript", type=str, help="Path to the input transcript.")
parser.add_argument("output_transcript", type=str, help="Path to the input transcript.")
args = parser.parse_args()

# Read TSV input
segments = []
with open(args.input_transcript, "r") as f:
    reader = csv.DictReader(f, delimiter="\t")
    for row in reader:
        segments.append(
            {
                "start": int(row["start"]),
                "end": int(row["end"]),
                "text": row["text"].strip(),
            }
        )

# Process merging
segments = split_on_periods(segments)
improved_segments = merge_transcript(segments, merge_on_punctuation)
# improved_segments = merge_transcript(improved_segments, merge_on_semantics)
#
improved_segments = improve_transcript_with_llm_v2(improved_segments)

# Output merged TSV
print("start\tend\ttext")
for seg in improved_segments:
    print(f"{seg['start']}\t{seg['end']}\t{seg['text']}")

# write the new transcript to a file
with open(args.output_transcript, "w") as f:
    writer = csv.DictWriter(f, fieldnames=["start", "end", "text"], delimiter="\t")
    writer.writeheader()
    for seg in improved_segments:
        writer.writerow(seg)

print(f"Transcript saved to {args.output_transcript}")

"""
TRANSCRIPT:
```
1:      Let's imagine you graduated from here and you foun
d a startup company and you're like, hey, my startup compa
ny offers some web service.
2:      Obviously, we need computers for that.
3:      Should I just go to AWS and rent some EC2 instance
s and maybe store data there?
4:      Or should I buy a bunch of machines and run the ma
chines myself maybe in a small warehouse like the one you 
saw in the video at the beginning?
5:      So in other words, should I use a public cloud lik
e AWS or Microsoft Azure or Google Cloud or any one of the
 other offerings out there?
6:      Or should I set up my own private cloud for my own
 company?
7:      So again, private cloud is accessible only to comp
any employees.
8:      Public cloud is accessible to anyone.
9:      So the choice between these two boils down to a bu
nch of different factors.
10:     What we will do over the next couple of slides is 
do a financial calculation to see which one is cheaper.
11:     As I said earlier in the first lecture, the answer
 to a lot of questions is time and money.
12:     So we will try to boil down this choice to just mo
ney itself.
13:     There are a lot of other factors involved in this 
decision as well, such as whether you can actually hire pe
ople or not if you want to run your own data center.
14:     We will ignore those aspects.
15:     We will focus purely on the money aspect, which on
e is cheaper.
16:     So single site cloud or single data center, should
 you outsource to a public cloud or should you own, meanin
g should you buy and manage those machines?
17:     So for this, we take a concrete example.
18:     The concrete example is the cloud computing testbe
d at UIUC, which was brought up in 2009 and ran for about 
four or five years.
19:     And of course, since then, it's been decommissione
d.
20:     But this is a fixed data point which gives us data
 points from both AWS prices back then but also invoices f
or the machine itself because I was involved in procuring 
this very large machine.
```
```
1: Let's imagine you graduated from here and you found a s
tartup company and you're like, hey, my startup company of
fers some web service.
2: Obviously, we need computers for that. Should I just go
 to AWS and rent some EC2 instances and maybe store data t
here?
3: Or should I buy a bunch of machines and run the machine
s myself, maybe in a small warehouse like the one you saw 
in the video at the beginning?
4: So in other words, should I use a public cloud like AWS
 or Microsoft Azure or Google Cloud or any one of the othe
r offerings out there? 
5: Or should I set up my own private cloud for my own comp
any?
6: So again, private cloud is accessible only to company e
mployees. Public cloud is accessible to anyone.
7: So the choice between these two boils down to a bunch o
f different factors.
8: What we will do over the next couple of slides is do a 
financial calculation to see which one is cheaper.
9: As I said earlier in the first lecture, the answer to a
 lot of questions is time and money.
10: So we will try to boil down this choice to just money 
itself.
11: There are a lot of other factors involved in this deci
sion as well, such as whether you can actually hire people
 or not if you want to run your own data center.
12: We will ignore those aspects. 
13: We will focus purely on the money aspect, which one is
 cheaper?
14: So single site cloud or single data center, should you
 outsource to a public cloud or should you own, meaning sh
ould you buy and manage those machines?
15: For this, we take a concrete example.
16: The concrete example is the cloud computing testbed at
 UIUC, which was brought up in 2009 and ran for about four
 or five years.
17: And of course, since then, it's been decommissioned. 
18: But this is a fixed data point which gives us data poi
nts from both AWS prices back then but also invoices for t
he machine itself because I was involved in procuring this
 very large machine.
```
Prompt:  Revise the lecture transcript adhering strictly t
o these rules:

1. **Edits MUST:**
   - Fix grammar/sentence structure WITHOUT changing origi
nal words
   - Keep ALL technical terms & phrases 
   - Remove filler words/phrases 
   - Merge lines ONLY when necessary for flow, using origi
nal wording
   - Preserve every number from original transcript - incl
ude blank lines for removed segments

2. **Format REQUIREMENTS:**
   - Input line numbers MUST be preserved in order (1-20)
   - Merged lines show combined numbers (1-2)
   - Output MUST have EXACTLY 20 lines matching input numb
ering

3. **Output FORBIDDEN from:**
   - Paraphrasing technical descriptions
   - Changing sentence order
   - Omitting numbers from original sequence

Respond ONLY with revised transcript between ```, NO comme
ntary.
    
TRANSCRIPT:
```
1:      As I said earlier in the first lecture, the answer
 to a lot of questions is time and money.
2:      So we will try to boil down this choice to just mo
ney itself.
3:      There are a lot of other factors involved in this 
decision as well, such as whether you can actually hire pe
ople or not if you want to run your own data center.
4:      We will ignore those aspects.
5:      We will focus purely on the money aspect, which on
e is cheaper.
6:      So single site cloud or single data center, should
 you outsource to a public cloud or should you own, meanin
g should you buy and manage those machines?
7:      So for this, we take a concrete example.
8:      The concrete example is the cloud computing testbe
d at UIUC, which was brought up in 2009 and ran for about 
four or five years.
9:      And of course, since then, it's been decommissione
d.
10:     But this is a fixed data point which gives us data
 points from both AWS prices back then but also invoices f
or the machine itself because I was involved in procuring 
this very large machine.
11:     We have actual numbers from the procurement of th
se machines which we can plug into our equations.
12:     So the cluster we bought back then had 128 server
 with each server having eight cores, so a total of 1,024
cores.
13:     And it had 524 terabytes of storage which is abou
 half a petabyte of storage.
14:     So let's imagine that you set up a startup compan
 and your startup company requires 1,024 VMs or 1,024 vir
ual CPUs and about half a petabyte of storage.
15:     How much does it cost to rent this on a public cl
ud?
16:     So we'll plug the storage amount, 524 terabytes i
to the S3 cost, simple storage service on AWS.
17:     So back then in 2009, for this much amount of dat
, the cost per gigabyte month was 12 cents, right?
18:     And so for storage, we'll multiply 12 cents, mult
ply it by 524, multiply it by 1,000.
19:     1,000 is needed because you have 1,000 gigabytes 
n a terabyte and the dollar cost of 12 cents is per gigab
te month.
20:     So the 62,000 is the cost for storing your half a
petabyte of data on S3 per month.
```
```
1: As I said earlier in the first lecture, the answer to 
a lot of questions is time and money.
2: So we will try to boil down this choice to just money 
itself.
3: There are many other factors involved in this decision
 as well, such as whether you can actually hire people or
 not if you want to run your own data center.
4: We will ignore those aspects.
5: We will focus purely on the money aspect, which one is
 cheaper.
6: So single site cloud or single data center, should you
 outsource to a public cloud or should you own, meaning s
hould you buy and manage those machines?
7: So for this, we take a concrete example.
8: The concrete example is the cloud computing testbed at
 UIUC, which was brought up in 2009 and ran for about fou
r or five years.
9: And of course, since then, it's been decommissioned.
10: But this is a fixed data point which gives us data po
ints from both AWS prices back then but also invoices for
 the machine itself because I was involved in procuring t
his very large machine.
11: We have actual numbers from the procurement of these 
machines which we can plug into our equations.
12: So the cluster we bought back then had 128 servers wi
th each server having eight cores, so a total of 1,024 co
res.
13: And it had 524 terabytes of storage which is about ha
lf a petabyte of storage.
14: Now let's imagine that you set up a startup company a
nd your startup company requires 1,024 VMs or 1,024 virtu
al CPUs and about half a petabyte of storage.
15: How much does it cost to rent this on a public cloud?
16: So we'll plug the storage amount into the S3 cost, si
mple storage service on AWS.
17: Back then in 2009, for this amount of data, the cost 
per gigabyte month was 12 cents, right?
18: And so for storage, we multiply 12 cents by 524, and 
by 1,000 because there are 1,000 gigabytes in a terabyte,
 resulting in a monthly cost of $62,000.
19:
20: So the cost for storing your half a petabyte of data 
on S3 per month is $62,000.
```
Prompt:  Revise the lecture transcript adhering strictly 
to these rules:

1. **Edits MUST:**
   - Fix grammar/sentence structure WITHOUT changing orig
inal words
   - Keep ALL technical terms & phrases 
   - Remove filler words/phrases 
   - Merge lines ONLY when necessary for flow, using orig
inal wording
   - Preserve every number from original transcript - inc
lude blank lines for removed segments

2. **Format REQUIREMENTS:**
   - Input line numbers MUST be preserved in order (1-20)
   - Merged lines show combined numbers (1-2)
   - Output MUST have EXACTLY 20 lines matching input num
bering

3. **Output FORBIDDEN from:**
   - Paraphrasing technical descriptions
   - Changing sentence order
   - Omitting numbers from original sequence

Respond ONLY with revised transcript between ```, NO comm
entary.
    
TRANSCRIPT:
```
1:      We have actual numbers from the procurement of th
ese machines which we can plug into our equations.
2:      So the cluster we bought back then had 128 server
s with each server having eight cores, so a total of 1,02
4 cores.
3:      And it had 524 terabytes of storage which is abou
t half a petabyte of storage.
4:      So let's imagine that you set up a startup compan
y and your startup company requires 1,024 VMs or 1,024 vi
rtual CPUs and about half a petabyte of storage.
5:      How much does it cost to rent this on a public cl
oud?
6:      So we'll plug the storage amount, 524 terabytes i
nto the S3 cost, simple storage service on AWS.
7:      So back then in 2009, for this much amount of dat
a, the cost per gigabyte month was 12 cents, right?
8:      And so for storage, we'll multiply 12 cents, mult
iply it by 524, multiply it by 1,000.
9:      1,000 is needed because you have 1,000 gigabytes 
in a terabyte and the dollar cost of 12 cents is per giga
byte month.
10:     So the 62,000 is the cost for storing your half a
 petabyte of data on S3 per month.
11:     So you'll keep incurring this cost every month th
at you store that data.
12:     Okay, so let's hold that to the side.
13:     Let's do a similar calculation for the compute it
self.
14:     So EC2 costs back then were about 10 cents per CP
U hour.
15:     This is, again, a cost from 2009.
16:     Nowadays, EC2 costs are much, much cheaper, but s
o that we are comparing apples with apples, we're going t
o be comparing EC2 costs from back then rather than from 
today.
17:     Yes?
18:     Which one was the approximate cost?
19:     That is an exact cost.
20:     Those are exact costs from 2009.
```
```
1: We have actual numbers from the procurement of these m
achines which we can plug into our equations.
2: So, the cluster we bought back then had 128 servers wi
th each server having eight cores, so a total of 1,024 co
res.
3: And it had 524 terabytes of storage, which is about ha
lf a petabyte of storage.
4: So let's imagine that you set up a startup company and
 your startup company requires 1,024 VMs or 1,024 virtual
 CPUs and about half a petabyte of storage.
5: How much does it cost to rent this on a public cloud?
6: So we'll plug the storage amount, 524 terabytes, into 
the S3 cost, simple storage service on AWS.
7: So back then in 2009, for this much amount of data, th
e cost per gigabyte month was 12 cents.
8: And so for storage, we'll multiply 12 cents by 524 and
 then multiply that result by 1,000 because there are 1,0
00 gigabytes in a terabyte, and the dollar cost is per gi
gabyte month.
9: The total cost for storing half a petabyte of data on 
S3 per month is $62,000.
10: This cost will be incurred every month that you store
 that data.
11: Okay, so let's hold that to the side.
12: Let's do a similar calculation for the compute itself
.
13: So EC2 costs back then were about 10 cents per CPU ho
ur.
14: Nowdays, EC2 costs are much cheaper, but to compare a
pples with apples, we're going to be comparing EC2 costs 
from back then rather than from today.
15: Which one was the approximate cost?
16: That is an exact cost.
17: Those are exact costs from 2009.
18:
19:
20:
```
Prompt:  Revise the lecture transcript adhering strictly 
to these rules:

1. **Edits MUST:**
   - Fix grammar/sentence structure WITHOUT changing orig
inal words
   - Keep ALL technical terms & phrases 
   - Remove filler words/phrases 
   - Merge lines ONLY when necessary for flow, using orig
inal wording
   - Preserve every number from original transcript - inc
lude blank lines for removed segments

2. **Format REQUIREMENTS:**
   - Input line numbers MUST be preserved in order (1-20)
   - Merged lines show combined numbers (1-2)
   - Output MUST have EXACTLY 20 lines matching input num
bering

3. **Output FORBIDDEN from:**
   - Paraphrasing technical descriptions
   - Changing sentence order
   - Omitting numbers from original sequence

Respond ONLY with revised transcript between ```, NO comm
entary.
    
TRANSCRIPT:
```
1:      So you'll keep incurring this cost every month th
at you store that data.
2:      Okay, so let's hold that to the side.
3:      Let's do a similar calculation for the compute it
self.
4:      So EC2 costs back then were about 10 cents per CP
U hour.
5:      This is, again, a cost from 2009.
6:      Nowadays, EC2 costs are much, much cheaper, but s
o that we are comparing apples with apples, we're going t
o be comparing EC2 costs from back then rather than from 
today.
7:      Yes?
8:      Which one was the approximate cost?
9:      That is an exact cost.
10:     Those are exact costs from 2009.
11:     Correct, yes.
12:     Even today, you get charged on EC2 for however ma
ny VMs you use for the hour.
13:     So if you rent, say, four VMs at 10 cents a piece
 and you use them for 15 minutes, you're going to be payi
ng for four VM hours still, 40 cents, no matter what frac
tion of the hour you use it.
14:     So it's always quantized.
15:     All right, so the cost of the compute, we have 1,
024 cores, there's 1,024 here, and remember we're calcula
ting the monthly cost.
16:     So there are 24 hours in a day, 24, and there's 3
0 days in a month.
17:     So the monthly cost is 10 cents multiplied by 1,0
24, cores multiplied by 24 hours multiplied by 30 days.
18:     That comes out to be, if you add up the storage c
ost, that comes to be 136K, 136,000 for the compute and t
he storage put together per month.
19:     So if you use a service for 12 months, the cost w
ill be this times 12.
20:     But for now, let's stick with the monthly costs.
```
```
1: So you'll keep incurring this cost every month that yo
u store that data.
2: Okay, so let's hold that to the side.
3: Let's do a similar calculation for the compute itself.
4: So EC2 costs back then were about 10 cents per CPU hou
r.
5: This is, again, a cost from 2009.
6: Nowdays, EC2 costs are much, much cheaper, but so that
 we are comparing apples with apples, we're going to be c
omparing EC2 costs from back then rather than from today.
7: Yes?
8: Which one was the approximate cost?
9: That is an exact cost.
10: Those are exact costs from 2009.
11: Correct, yes.
12: Even today, you get charged on EC2 for however many V
Ms you use for the hour.
13: So if you rent, say, four VMs at 10 cents a piece and
 you use them for 15 minutes, you're going to be paying f
or four VM hours still, 40 cents, no matter what fraction
 of the hour you use it.
14: So it's always quantized.
15: All right, so the cost of the compute, we have 1,024 
cores, there's 1,024 here, and remember we're calculating
 the monthly cost.
16: So there are 24 hours in a day, 24, and there's 30 da
ys in a month.
17: So the monthly cost is 10 cents multiplied by 1,024 c
ores, multiplied by 24 hours, multiplied by 30 days.
18: That comes out to be, if you add up the storage cost,
 that comes to be $136,000 for the compute and the storag
e put together per month.
19: So if you use a service for 12 months, the cost will 
be this times 12.
20: But for now, let's stick with the monthly costs.
```
Prompt:  Revise the lecture transcript adhering strictly 
to these rules:

1. **Edits MUST:**
   - Fix grammar/sentence structure WITHOUT changing orig
inal words
   - Keep ALL technical terms & phrases 
   - Remove filler words/phrases 
   - Merge lines ONLY when necessary for flow, using orig
inal wording
   - Preserve every number from original transcript - inc
lude blank lines for removed segments

2. **Format REQUIREMENTS:**
   - Input line numbers MUST be preserved in order (1-20)
   - Merged lines show combined numbers (1-2)
   - Output MUST have EXACTLY 20 lines matching input num
bering

3. **Output FORBIDDEN from:**
   - Paraphrasing technical descriptions
   - Changing sentence order
   - Omitting numbers from original sequence

Respond ONLY with revised transcript between ```, NO comm
entary.
    
TRANSCRIPT:
```
1:      Correct, yes.
2:      Even today, you get charged on EC2 for however ma
ny VMs you use for the hour.
3:      So if you rent, say, four VMs at 10 cents a piece
 and you use them for 15 minutes, you're going to be payi
ng for four VM hours still, 40 cents, no matter what frac
tion of the hour you use it.
4:      So it's always quantized.
5:      All right, so the cost of the compute, we have 1,
024 cores, there's 1,024 here, and remember we're calcula
ting the monthly cost.
6:      So there are 24 hours in a day, 24, and there's 3
0 days in a month.
7:      So the monthly cost is 10 cents multiplied by 1,0
24, cores multiplied by 24 hours multiplied by 30 days.
8:      That comes out to be, if you add up the storage c
ost, that comes to be 136K, 136,000 for the compute and t
he storage put together per month.
9:      So if you use a service for 12 months, the cost w
ill be this times 12.
10:     But for now, let's stick with the monthly costs.
11:     Those are very large numbers, right?
12:     So maybe owning it is going to be cheaper, but no
w let's plug in the actual numbers for the storage procur
ement.
13:     So we spend for the storage, the disks itself, we
 spent about 350K, right?
14:     And let's say you know how long your startup is g
oing to survive.
15:     Of course, you don't know that.
16:     But let's say you say, okay, I know my startup is
 going to survive for 12 months, right?
17:     So we plug in the value of M equals 12 here, and 
that gives you 349K divided by M gives you the monthly co
st of procuring the storage itself.
18:     Same thing we can do for the total cost.
19:     The total cost was about 1.5 million, and so that
 plugged in divided by M gives you the monthly cost of th
e total cluster itself, compute and storage and everythin
g else put together.
20:     Now there's some hidden numbers in here, which I'
ll describe to you momentarily.
```

"""
