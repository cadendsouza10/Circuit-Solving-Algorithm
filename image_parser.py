import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr
import re


################################################
# (1) Load Image
################################################

IMAGE_PATH = "images/test_circuit1.png"

image = cv2.imread(IMAGE_PATH)

if image is None:
    raise FileNotFoundError(
        f"Could not read image: {IMAGE_PATH}"
    )

gray = cv2.cvtColor(
    image,
    cv2.COLOR_BGR2GRAY
)

# IMPORTANT:
# Define image dimensions early
height, width = gray.shape


_, binary = cv2.threshold(
    gray,
    180,
    255,
    cv2.THRESH_BINARY_INV
)


################################################
# (2) Display Binary Image
################################################

plt.imshow(
    binary,
    cmap="gray"
)

plt.title("Binary Circuit")
plt.axis("off")
plt.show()


################################################
# (3) OCR
################################################

reader = easyocr.Reader(["en"])

results = reader.readtext(image)


################################################
# (4) OCR Helper Functions
################################################

def clean_ocr_text(text):

    text = text.strip().upper()

    replacements = {
        "RL": "R1",
        "RI": "R1",
        "R|": "R1",
        "RZ": "R2"
    }

    return replacements.get(
        text,
        text
    )


def extract_number(text):

    text = text.replace(" ", "")

    match = re.search(
        r"\d+(?:\.\d+)?",
        text
    )

    if match is None:
        return None

    try:
        return float(
            match.group()
        )

    except ValueError:
        return None


def parse_value(text):

    text = text.strip()
    text = text.replace(" ", "")

    multipliers = {
        "k": 1e3,
        "K": 1e3,
        "M": 1e6,
        "m": 1e-3,
        "u": 1e-6,
        "n": 1e-9,
        "p": 1e-12
    }

    text = text.replace("Ω", "")
    text = text.replace("OHM", "")
    text = text.replace("ohm", "")

    text = text.replace("V", "")
    text = text.replace("v", "")

    if not text:
        return None

    try:

        last_character = text[-1]

        if last_character in multipliers:

            number = float(
                text[:-1]
            )

            return (
                number
                *
                multipliers[
                    last_character
                ]
            )

        return float(text)

    except ValueError:

        return extract_number(text)


################################################
# (5) Store OCR Results + Positions
################################################

ocr_items = []

for result in results:

    bounding_box = result[0]
    raw_text = result[1]
    confidence = result[2]

    text = clean_ocr_text(
        raw_text
    )

    xs = [
        point[0]
        for point in bounding_box
    ]

    ys = [
        point[1]
        for point in bounding_box
    ]

    center_x = float(
        sum(xs) / len(xs)
    )

    center_y = float(
        sum(ys) / len(ys)
    )

    ocr_items.append({
        "text": text,
        "raw_text": raw_text,
        "confidence": confidence,
        "center": (
            center_x,
            center_y
        ),
        "box": bounding_box
    })


print("\nTEXT DETECTED")
print("--------------------------------")

for item in ocr_items:

    print(
        f"{item['text']:<25} "
        f"confidence="
        f"{item['confidence']:.2f}"
    )


print("\nOCR ITEMS WITH POSITION")
print("--------------------------------")

for item in ocr_items:

    print(
        f"{item['text']:<15} "
        f"at "
        f"({item['center'][0]:.1f}, "
        f"{item['center'][1]:.1f})"
    )


################################################
# (6) Detect Resistor Values
################################################

resistors = []

for item in ocr_items:

    text = item["text"]

    cx, cy = item["center"]


    # Ignore title
    if "TEST IMAGE" in text:
        continue


    # ignore voltage text
    if "V" in text.upper():
        continue


    value = parse_value(text)

    if value is None:
        continue


    if cx < width * 0.50:
        continue


    resistors.append({
        "value": value,
        "center": item["center"],
        "ocr_box": item["box"]
    })


# top to bottom
resistors.sort(
    key=lambda resistor:
    resistor["center"][1]
)


# R1, R2, ...
for index, resistor in enumerate(
    resistors,
    start=1
):

    resistor["name"] = (
        f"R{index}"
    )


print("\nCOMPONENT LABELS")
print("--------------------------------")

for resistor in resistors:

    print(
        f"{resistor['name']} "
        f"-> resistor "
        f"value="
        f"{resistor['value']} ohms "
        f"at "
        f"({resistor['center'][0]:.1f}, "
        f"{resistor['center'][1]:.1f})"
    )


if len(resistors) == 0:

    raise RuntimeError(
        "No resistor values were detected."
    )


################################################
# (7) Find Actual Resistor Symbol Regions
################################################

resistor_regions = []


for resistor in resistors:

    cx, cy = resistor["center"]

    cx = int(cx)
    cy = int(cy)


    # Search LEFT of resistor value text for zigzag
    search_x1 = max(
        0,
        cx - 260
    )

    search_x2 = max(
        0,
        cx - 60
    )

    search_y1 = max(
        0,
        cy - 100
    )

    search_y2 = min(
        height,
        cy + 100
    )


    roi = binary[
        search_y1:search_y2,
        search_x1:search_x2
    ]


    contours, _ = cv2.findContours(
        roi,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )


    best_box = None
    best_score = -float("inf")


    for contour in contours:

        x, y, w, h = (
            cv2.boundingRect(
                contour
            )
        )


        # ignore small things
        if w < 5 or h < 15:
            continue


        # Current test circuit --> resistors are vertical.
        score = (
            h
            -
            0.5 * w
        )


        if score > best_score:

            best_score = score

            best_box = (
                search_x1 + x,
                search_y1 + y,
                search_x1 + x + w,
                search_y1 + y + h
            )


    if best_box is None:

        best_box = (
            max(
                0,
                cx - 180
            ),
            max(
                0,
                cy - 70
            ),
            max(
                0,
                cx - 100
            ),
            min(
                height,
                cy + 70
            )
        )


    resistor_regions.append({
        "name":
            resistor["name"],

        "value":
            resistor["value"],

        "box":
            best_box
    })


################################################
# (8) Display Detected Component Regions
################################################

component_debug = image.copy()


for resistor in resistor_regions:

    x1, y1, x2, y2 = (
        resistor["box"]
    )

    cv2.rectangle(
        component_debug,
        (x1, y1),
        (x2, y2),
        (255, 0, 0),
        2
    )

    cv2.putText(
        component_debug,
        resistor["name"],
        (
            x1,
            max(
                0,
                y1 - 5
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (255, 0, 0),
        2
    )


plt.imshow(
    cv2.cvtColor(
        component_debug,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "Detected Component Regions"
)

plt.axis("off")
plt.show()


################################################
# (9) Mask Components Before Wire Detection
################################################

wire_binary = binary.copy()


# Remove resistor symbols
for resistor in resistor_regions:

    x1, y1, x2, y2 = (
        resistor["box"]
    )

    pad = 5

    cv2.rectangle(
        wire_binary,

        (
            max(
                0,
                x1 - pad
            ),
            max(
                0,
                y1 - pad
            )
        ),

        (
            min(
                width - 1,
                x2 + pad
            ),
            min(
                height - 1,
                y2 + pad
            )
        ),

        0,
        -1
    )


# Remove OCR text
for item in ocr_items:

    box = np.array(
        item["box"],
        dtype=np.int32
    )

    cv2.fillPoly(
        wire_binary,
        [box],
        0
    )


plt.imshow(
    wire_binary,
    cmap="gray"
)

plt.title(
    "Circuit With Components Masked"
)

plt.axis("off")
plt.show()


################################################
# (10) Detect Clean Wires
################################################

clean_lines = cv2.HoughLinesP(
    wire_binary,
    rho=1,
    theta=np.pi / 180,
    threshold=35,
    minLineLength=25,
    maxLineGap=8
)


clean_wire_segments = []


if clean_lines is not None:

    for line in clean_lines:

        x1, y1, x2, y2 = (
            np.array(
                line
            ).flatten()
        )

        clean_wire_segments.append({

            "start": (
                int(x1),
                int(y1)
            ),

            "end": (
                int(x2),
                int(y2)
            )

        })


################################################
# (11) Classify Horizontal / Vertical Wires
################################################

horizontal_wires = []
vertical_wires = []

ANGLE_TOLERANCE = 10


for wire in clean_wire_segments:

    x1, y1 = wire["start"]
    x2, y2 = wire["end"]

    dx = x2 - x1
    dy = y2 - y1

    angle = abs(
        np.degrees(
            np.arctan2(
                dy,
                dx
            )
        )
    )


    # Horizontal
    if (
        angle
        <
        ANGLE_TOLERANCE
        or
        angle
        >
        180 - ANGLE_TOLERANCE
    ):

        horizontal_wires.append(
            wire
        )


    # Vertical
    elif (
        abs(
            angle - 90
        )
        <
        ANGLE_TOLERANCE
    ):

        vertical_wires.append(
            wire
        )


print("\nWIRE ORIENTATION")
print("--------------------------------")

print(
    "Horizontal wires:",
    len(horizontal_wires)
)

print(
    "Vertical wires:",
    len(vertical_wires)
)


################################################
# (12) Merge Similar Horizontal Wires
################################################

def merge_horizontal_wires(
    wires,
    y_tolerance=8,
    gap_tolerance=15
):

    normalized = []


    for wire in wires:

        x1, y1 = wire["start"]
        x2, y2 = wire["end"]


        if x1 > x2:

            x1, x2 = x2, x1
            y1, y2 = y2, y1


        y = int(
            (y1 + y2)
            /
            2
        )


        normalized.append(
            [
                x1,
                x2,
                y
            ]
        )


    normalized.sort(
        key=lambda wire:
        (
            wire[2],
            wire[0]
        )
    )


    merged = []


    for current in normalized:

        cx1, cx2, cy = current

        matched = False


        for existing in merged:

            ex1, ex2, ey = existing


            same_y = (
                abs(
                    cy - ey
                )
                <=
                y_tolerance
            )


            overlaps = (
                cx1
                <=
                ex2
                +
                gap_tolerance
                and
                cx2
                >=
                ex1
                -
                gap_tolerance
            )


            if (
                same_y
                and
                overlaps
            ):

                existing[0] = min(
                    ex1,
                    cx1
                )

                existing[1] = max(
                    ex2,
                    cx2
                )

                existing[2] = int(
                    (
                        ey + cy
                    )
                    /
                    2
                )

                matched = True

                break


        if not matched:

            merged.append(
                [
                    cx1,
                    cx2,
                    cy
                ]
            )


    return [

        {
            "start":
                (
                    x1,
                    y
                ),

            "end":
                (
                    x2,
                    y
                )
        }

        for x1, x2, y in merged
    ]


################################################
# (13) Merge Similar Vertical Wires
################################################

def merge_vertical_wires(
    wires,
    x_tolerance=8,
    gap_tolerance=15
):

    normalized = []


    for wire in wires:

        x1, y1 = wire["start"]
        x2, y2 = wire["end"]


        if y1 > y2:

            y1, y2 = y2, y1
            x1, x2 = x2, x1


        x = int(
            (x1 + x2)
            /
            2
        )


        normalized.append(
            [
                y1,
                y2,
                x
            ]
        )


    normalized.sort(
        key=lambda wire:
        (
            wire[2],
            wire[0]
        )
    )


    merged = []


    for current in normalized:

        cy1, cy2, cx = current

        matched = False


        for existing in merged:

            ey1, ey2, ex = existing


            same_x = (
                abs(
                    cx - ex
                )
                <=
                x_tolerance
            )


            overlaps = (
                cy1
                <=
                ey2
                +
                gap_tolerance
                and
                cy2
                >=
                ey1
                -
                gap_tolerance
            )


            if (
                same_x
                and
                overlaps
            ):

                existing[0] = min(
                    ey1,
                    cy1
                )

                existing[1] = max(
                    ey2,
                    cy2
                )

                existing[2] = int(
                    (
                        ex + cx
                    )
                    /
                    2
                )

                matched = True

                break


        if not matched:

            merged.append(
                [
                    cy1,
                    cy2,
                    cx
                ]
            )


    return [

        {
            "start":
                (
                    x,
                    y1
                ),

            "end":
                (
                    x,
                    y2
                )
        }

        for y1, y2, x in merged
    ]


################################################
# (14) Final Clean Wire List
################################################

merged_horizontal = (
    merge_horizontal_wires(
        horizontal_wires
    )
)

merged_vertical = (
    merge_vertical_wires(
        vertical_wires
    )
)


clean_wires = (
    merged_horizontal
    +
    merged_vertical
)


print("\nMERGED WIRES")
print("--------------------------------")

for index, wire in enumerate(
    clean_wires
):

    print(
        f"Wire {index}: "
        f"{wire['start']} "
        f"-> "
        f"{wire['end']}"
    )


################################################
# (15) Display Clean Wires
################################################

wire_debug = image.copy()


for wire in clean_wires:

    cv2.line(
        wire_debug,
        wire["start"],
        wire["end"],
        (0, 255, 0),
        3
    )


plt.imshow(
    cv2.cvtColor(
        wire_debug,
        cv2.COLOR_BGR2RGB
    )
)

plt.title(
    "Clean Wire Detection"
)

plt.axis("off")
plt.show()


################################################
# (16) Estimate Resistor Terminals
################################################

resistor_terminals = []


for resistor in resistor_regions:

    x1, y1, x2, y2 = (
        resistor["box"]
    )


    center_x = int(
        (
            x1 + x2
        )
        /
        2
    )


    top_terminal = (
        center_x,
        y1
    )


    bottom_terminal = (
        center_x,
        y2
    )


    resistor_terminals.append({

        "name":
            resistor["name"],

        "value":
            resistor["value"],

        "top_terminal":
            top_terminal,

        "bottom_terminal":
            bottom_terminal

    })


print(
    "\nESTIMATED RESISTOR TERMINALS"
)

print(
    "--------------------------------"
)


for resistor in resistor_terminals:

    print(
        f"{resistor['name']} "
        f"top="
        f"{resistor['top_terminal']} "
        f"bottom="
        f"{resistor['bottom_terminal']}"
    )


################################################
# (17) Distance Functions
################################################

def point_to_segment_distance(
    point,
    start,
    end
):

    px, py = point

    x1, y1 = start
    x2, y2 = end

    dx = x2 - x1
    dy = y2 - y1


    if (
        dx == 0
        and
        dy == 0
    ):

        return np.hypot(
            px - x1,
            py - y1
        )


    t = (
        (
            px - x1
        )
        *
        dx
        +
        (
            py - y1
        )
        *
        dy
    ) / (
        dx * dx
        +
        dy * dy
    )


    t = max(
        0,
        min(
            1,
            t
        )
    )


    nearest_x = (
        x1
        +
        t * dx
    )


    nearest_y = (
        y1
        +
        t * dy
    )


    return np.hypot(
        px - nearest_x,
        py - nearest_y
    )


def find_nearest_wire(
    point,
    wires
):

    best_index = None
    best_distance = float(
        "inf"
    )


    for index, wire in enumerate(
        wires
    ):

        distance = (
            point_to_segment_distance(
                point,
                wire["start"],
                wire["end"]
            )
        )


        if distance < best_distance:

            best_distance = distance
            best_index = index


    return (
        best_index,
        best_distance
    )


################################################
# (18) Match Resistors to Wires
################################################

resistor_connections = []


print(
    "\nTERMINAL -> WIRE MATCHING"
)

print(
    "--------------------------------"
)


for resistor in resistor_terminals:

    top_wire, top_distance = (
        find_nearest_wire(
            resistor[
                "top_terminal"
            ],
            clean_wires
        )
    )


    bottom_wire, bottom_distance = (
        find_nearest_wire(
            resistor[
                "bottom_terminal"
            ],
            clean_wires
        )
    )


    resistor_connections.append({

        "name":
            resistor["name"],

        "value":
            resistor["value"],

        "wire1":
            top_wire,

        "wire2":
            bottom_wire

    })


    print(
        f"{resistor['name']} top "
        f"-> Wire {top_wire}, "
        f"distance="
        f"{top_distance:.1f}"
    )


    print(
        f"{resistor['name']} bottom "
        f"-> Wire {bottom_wire}, "
        f"distance="
        f"{bottom_distance:.1f}"
    )


################################################
# (19) Group Connected Wire Segments
################################################

CONNECTION_TOLERANCE = 15


def orientation(p, q, r):

    value = (
        (q[1] - p[1]) * (r[0] - q[0])
        -
        (q[0] - p[0]) * (r[1] - q[1])
    )

    if abs(value) < 1e-9:
        return 0

    return 1 if value > 0 else 2


def on_segment(p, q, r, tolerance=CONNECTION_TOLERANCE):

    return (
        min(p[0], r[0]) - tolerance
        <= q[0] <=
        max(p[0], r[0]) + tolerance
        and
        min(p[1], r[1]) - tolerance
        <= q[1] <=
        max(p[1], r[1]) + tolerance
    )


def segments_intersect(
    p1,
    q1,
    p2,
    q2
):

    o1 = orientation(
        p1,
        q1,
        p2
    )

    o2 = orientation(
        p1,
        q1,
        q2
    )

    o3 = orientation(
        p2,
        q2,
        p1
    )

    o4 = orientation(
        p2,
        q2,
        q1
    )


    # gen. intersection
    if (
        o1 != o2
        and
        o3 != o4
    ):
        return True


    # collinear
    if (
        o1 == 0
        and
        on_segment(
            p1,
            p2,
            q1
        )
    ):
        return True

    if (
        o2 == 0
        and
        on_segment(
            p1,
            q2,
            q1
        )
    ):
        return True

    if (
        o3 == 0
        and
        on_segment(
            p2,
            p1,
            q2
        )
    ):
        return True

    if (
        o4 == 0
        and
        on_segment(
            p2,
            q1,
            q2
        )
    ):
        return True


    return False


def wires_are_connected(
    wire_a,
    wire_b
):

    a_start = wire_a["start"]
    a_end = wire_a["end"]

    b_start = wire_b["start"]
    b_end = wire_b["end"]


    # check actual intersect
    if segments_intersect(
        a_start,
        a_end,
        b_start,
        b_end
    ):
        return True


    # allow small gaps
    distances = [

        point_to_segment_distance(
            a_start,
            b_start,
            b_end
        ),

        point_to_segment_distance(
            a_end,
            b_start,
            b_end
        ),

        point_to_segment_distance(
            b_start,
            a_start,
            a_end
        ),

        point_to_segment_distance(
            b_end,
            a_start,
            a_end
        )

    ]


    return (
        min(distances)
        <=
        CONNECTION_TOLERANCE
    )


################################################
# (20) Union-Find
################################################

parent = list(
    range(
        len(clean_wires)
    )
)


def find(x):

    if parent[x] != x:

        parent[x] = find(
            parent[x]
        )

    return parent[x]


def union(a, b):

    root_a = find(a)
    root_b = find(b)

    if root_a != root_b:

        parent[root_b] = (
            root_a
        )


for i in range(
    len(clean_wires)
):

    for j in range(
        i + 1,
        len(clean_wires)
    ):

        if wires_are_connected(
            clean_wires[i],
            clean_wires[j]
        ):

            union(
                i,
                j
            )


################################################
# (21) Build Electrical Wire Groups
################################################

wire_groups = {}


for wire_index in range(
    len(clean_wires)
):

    root = find(
        wire_index
    )

    if root not in wire_groups:

        wire_groups[root] = []


    wire_groups[root].append(
        wire_index
    )


print(
    "\nELECTRICAL WIRE GROUPS"
)

print(
    "--------------------------------"
)


for group_id, wire_ids in (
    wire_groups.items()
):

    print(
        f"Group {group_id}: "
        f"Wires {wire_ids}"
    )


################################################
# (22) Wire -> Group Mapping
################################################

wire_to_group = {}


for group_id, wire_ids in (
    wire_groups.items()
):

    for wire_id in wire_ids:

        wire_to_group[
            wire_id
        ] = group_id


################################################
# (23) Find Top + Bottom Circuit Wires
################################################

horizontal_candidates = []


for index, wire in enumerate(
    clean_wires
):

    x1, y1 = wire["start"]
    x2, y2 = wire["end"]


    if abs(
        y1 - y2
    ) <= 5:

        length = abs(
            x2 - x1
        )


        # ignore battery plate
        if (
            length
            >
            width * 0.30
        ):

            horizontal_candidates.append({

                "wire":
                    index,

                "y":
                    int(
                        (
                            y1 + y2
                        )
                        /
                        2
                    ),

                "length":
                    length

            })


if not horizontal_candidates:

    raise RuntimeError(
        "Could not find top/bottom circuit wires."
    )


top_wire = min(
    horizontal_candidates,
    key=lambda item:
    item["y"]
)["wire"]


bottom_wire = max(
    horizontal_candidates,
    key=lambda item:
    item["y"]
)["wire"]


top_group = (
    wire_to_group[
        top_wire
    ]
)


ground_group = (
    wire_to_group[
        bottom_wire
    ]
)


print(
    "\nSOURCE CONNECTIONS"
)

print(
    "--------------------------------"
)

print(
    f"Top wire: "
    f"{top_wire} "
    f"-> Group "
    f"{top_group}"
)

print(
    f"Bottom wire: "
    f"{bottom_wire} "
    f"-> Group "
    f"{ground_group}"
)


################################################
# (24) Convert Resistor Wires -> Groups
################################################

for resistor in resistor_connections:

    resistor["group1"] = (
        wire_to_group[
            resistor["wire1"]
        ]
    )

    resistor["group2"] = (
        wire_to_group[
            resistor["wire2"]
        ]
    )


print(
    "\nRESISTOR GROUP CONNECTIONS"
)

print(
    "--------------------------------"
)


for resistor in resistor_connections:

    print(
        f"{resistor['name']}: "
        f"Group "
        f"{resistor['group1']} "
        f"-> Group "
        f"{resistor['group2']}"
    )


################################################
# (25) Assign Electrical Node Numbers
################################################

group_to_node = {}


# Ground = Node 0
group_to_node[
    ground_group
] = 0


# Positive source = Node 1
if top_group != ground_group:

    group_to_node[
        top_group
    ] = 1


next_node = 2


for resistor in resistor_connections:

    for group in [

        resistor["group1"],
        resistor["group2"]

    ]:

        if (
            group
            not in
            group_to_node
        ):

            group_to_node[
                group
            ] = next_node

            next_node += 1


print("\nNODE MAP")
print("--------------------------------")


for group, node in (
    group_to_node.items()
):

    print(
        f"Group {group} "
        f"-> Node {node}"
    )


################################################
# (26) Detect Voltage Source Value
################################################

voltage_value = None


# Attempt 1:
# Look for text explicitly containing V.
for item in ocr_items:

    text = item["text"]

    if "V" in text.upper():

        number = extract_number(
            text
        )

        if number is not None:

            voltage_value = number

            break


# Attempt 2:
# OCR may detect "9 V" as "'9".
if voltage_value is None:

    voltage_candidates = []


    for item in ocr_items:

        cx, cy = (
            item["center"]
        )


        # ignore right side
        if cx >= width * 0.50:
            continue


        # ignore title
        if (
            "TEST IMAGE"
            in
            item["text"]
        ):

            continue


        number = extract_number(
            item["text"]
        )


        if number is None:
            continue


        voltage_candidates.append({

            "value":
                number,

            "center":
                item["center"],

            "text":
                item["text"]

        })


    if voltage_candidates:

        # Pick candidate closest to vertical center
        best_candidate = min(

            voltage_candidates,

            key=lambda item:
            abs(
                item["center"][1]
                -
                height / 2
            )

        )


        voltage_value = (
            best_candidate[
                "value"
            ]
        )


if voltage_value is None:

    raise RuntimeError(
        "Could not detect voltage source value."
    )


print("\nVOLTAGE SOURCE")
print("--------------------------------")

print(
    f"Detected voltage: "
    f"{voltage_value} V"
)


################################################
# (27) Generate Final Netlist
################################################

netlist_lines = []


# volt source
netlist_lines.append(

    f"V1 "
    f"{group_to_node[top_group]} "
    f"{group_to_node[ground_group]} "
    f"{voltage_value}"

)


# resistors
for resistor in resistor_connections:

    node1 = group_to_node[
        resistor["group1"]
    ]

    node2 = group_to_node[
        resistor["group2"]
    ]


    value = (
        resistor["value"]
    )


    netlist_lines.append(

        f"{resistor['name']} "
        f"{node1} "
        f"{node2} "
        f"{value}"

    )


################################################
# (28) Print Final Netlist
################################################

print(
    "\nGENERATED NETLIST"
)

print(
    "--------------------------------"
)


for line in netlist_lines:

    print(line)


################################################
# (29) Write circuit.txt
################################################

with open(
    "circuit.txt",
    "w"
) as file:

    for line in netlist_lines:

        file.write(
            line
            +
            "\n"
        )


print(
    "\ncircuit.txt generated successfully."
)
