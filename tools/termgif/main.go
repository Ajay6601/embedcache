// termgif renders an animated terminal session to a GIF for the README and
// docs site — reproducible from code, no screen recording involved.
//
// It is a separate Go module so the main embedcache module stays
// dependency-free (this tool needs golang.org/x/image for font rendering).
//
// Usage: go run . -out ../../docs/assets/demo.gif
package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"image/gif"
	"os"
	"strings"

	"golang.org/x/image/font"
	"golang.org/x/image/font/gofont/gomono"
	"golang.org/x/image/font/opentype"
	"golang.org/x/image/math/fixed"
)

var (
	outPath    = flag.String("out", "demo.gif", "output gif path")
	transcript = flag.String("transcript", "", "JSON transcript captured from a real run; if set, renders that instead of the built-in demo")
	titleText  = flag.String("title", "embedcache", "title-bar caption")
	widthFlag  = flag.Int("width", 760, "gif width in px")
	heightFlag = flag.Int("height", 460, "gif height in px")
)

var (
	bg    = color.RGBA{0x0b, 0x0f, 0x14, 0xff}
	barBg = color.RGBA{0x11, 0x18, 0x23, 0xff}
	text  = color.RGBA{0xd7, 0xe0, 0xea, 0xff}
	dim   = color.RGBA{0x5c, 0x69, 0x75, 0xff}
	green = color.RGBA{0x34, 0xd3, 0x99, 0xff}
	mint  = color.RGBA{0x6e, 0xe7, 0xb7, 0xff}
	amber = color.RGBA{0xfb, 0xbf, 0x24, 0xff}
	cyan  = color.RGBA{0x67, 0xe8, 0xf9, 0xff}
	red   = color.RGBA{0xff, 0x5f, 0x57, 0xff}
	ylw   = color.RGBA{0xfe, 0xbc, 0x2e, 0xff}
	grn2  = color.RGBA{0x28, 0xc8, 0x40, 0xff}
)

var palette = color.Palette{bg, barBg, text, dim, green, mint, amber, cyan, red, ylw, grn2}

var (
	width  = 760
	height = 460
)

const (
	marginX = 22
	topY    = 58 // below the title bar
	lineH   = 19
	fps     = 10 // 100ms per frame
)

type span struct {
	text string
	col  color.RGBA
}
type row []span

// wrapRow splits a row into as many visual lines as it takes to fit maxChars,
// breaking at spaces where possible, the way a real terminal soft-wraps. Colors
// are preserved across the break.
func wrapRow(r row, maxChars int) []row {
	type rc struct {
		ch  rune
		col color.RGBA
	}
	var flat []rc
	for _, s := range r {
		for _, ch := range s.text {
			flat = append(flat, rc{ch, s.col})
		}
	}
	if maxChars <= 0 || len(flat) <= maxChars {
		return []row{r}
	}
	var out []row
	for start := 0; start < len(flat); {
		end := start + maxChars
		if end >= len(flat) {
			end = len(flat)
		} else {
			// prefer breaking at a space in the last third of the line
			for b := end; b > start+maxChars*2/3; b-- {
				if flat[b-1].ch == ' ' {
					end = b
					break
				}
			}
		}
		var rr row
		for i := start; i < end; i++ {
			if len(rr) > 0 && rr[len(rr)-1].col == flat[i].col {
				rr[len(rr)-1].text += string(flat[i].ch)
			} else {
				rr = append(rr, span{string(flat[i].ch), flat[i].col})
			}
		}
		out = append(out, rr)
		start = end
	}
	return out
}

// a step is one line of the session, rendered from a real captured transcript.
type step struct {
	r      row
	typing bool // reveal char by char
	pause  int  // extra hold frames after the row completes
}

// transcriptEvent is one line captured from a real embedcache run. The capture
// harness (tools/gifcap) actually executes the commands against real backends
// and records their real output into these events; termgif only replays them.
type transcriptEvent struct {
	Kind  string `json:"kind"`            // cmd | out | dim
	Text  string `json:"text"`            // the literal line captured
	Color string `json:"color,omitempty"` // optional emphasis: green|mint|amber|cyan|red|dim
	Pause int    `json:"pause,omitempty"` // extra hold frames after the line
}

func colorByName(name string) color.RGBA {
	switch name {
	case "green":
		return green
	case "mint":
		return mint
	case "amber":
		return amber
	case "cyan":
		return cyan
	case "red":
		return red
	case "dim":
		return dim
	default:
		return text
	}
}

func scriptFromTranscript(path string) []step {
	raw, err := os.ReadFile(path)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	var events []transcriptEvent
	if err := json.Unmarshal(raw, &events); err != nil {
		fmt.Fprintln(os.Stderr, "parsing transcript:", err)
		os.Exit(1)
	}
	var steps []step
	for _, e := range events {
		var r row
		switch e.Kind {
		case "cmd":
			r = row{{"$ ", green}, {e.Text, text}}
			steps = append(steps, step{r: r, typing: true, pause: e.Pause})
			continue
		case "dim":
			r = row{{e.Text, dim}}
		default: // out
			// allow a colored prefix marker like "hit:" / "miss:" by coloring the
			// whole captured line when a color is given
			r = row{{e.Text, colorByName(e.Color)}}
		}
		steps = append(steps, step{r: r, pause: e.Pause})
	}
	return steps
}

func main() {
	flag.Parse()
	if *transcript == "" {
		fmt.Fprintln(os.Stderr, "termgif renders a real captured transcript; pass -transcript <file.json>")
		fmt.Fprintln(os.Stderr, "capture one with e.g.: go run ./experiments/gifcap -scenario serve -upstream http://localhost:11434 -out serve.json")
		os.Exit(2)
	}
	width, height = *widthFlag, *heightFlag
	face := loadFace()
	maxChars := (width - 2*marginX) / textWidth(face, "M")

	var frames []*image.Paletted
	var rows []row

	render := func(partial *row) {
		frames = append(frames, drawFrame(face, rows, partial, maxChars))
	}

	steps := scriptFromTranscript(*transcript)
	for _, st := range steps {
		if st.typing {
			// reveal the command 3 chars per frame
			full := st.r[len(st.r)-1].text
			for n := 0; n <= len(full); n += 3 {
				if n > len(full) {
					n = len(full)
				}
				p := row{st.r[0], {full[:min(n, len(full))], text}}
				render(&p)
			}
		}
		rows = append(rows, st.r)
		render(nil)
		for i := 0; i < st.pause; i++ {
			render(nil)
		}
		// scroll if we run out of vertical space, counting wrapped visual lines
		maxRows := (height - topY - 14) / lineH
		for visualLines(rows, maxChars) > maxRows && len(rows) > 1 {
			rows = rows[1:]
		}
	}
	// final hold
	for i := 0; i < 20; i++ {
		render(nil)
	}

	g := &gif.GIF{LoopCount: 0}
	for i, f := range frames {
		delay := 100 / (1000 / fps / 10) // 10 = 100ms in GIF centiseconds
		_ = i
		g.Image = append(g.Image, f)
		g.Delay = append(g.Delay, delay)
	}

	fh, err := os.Create(*outPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	defer fh.Close()
	if err := gif.EncodeAll(fh, g); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	fmt.Printf("wrote %s: %d frames, %dx%d\n", *outPath, len(frames), width, height)
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

func loadFace() font.Face {
	f, err := opentype.Parse(gomono.TTF)
	if err != nil {
		panic(err)
	}
	face, err := opentype.NewFace(f, &opentype.FaceOptions{Size: 13, DPI: 72, Hinting: font.HintingFull})
	if err != nil {
		panic(err)
	}
	return face
}

func visualLines(rows []row, maxChars int) int {
	n := 0
	for _, r := range rows {
		n += len(wrapRow(r, maxChars))
	}
	return n
}

func drawFrame(face font.Face, rows []row, partial *row, maxChars int) *image.Paletted {
	img := image.NewPaletted(image.Rect(0, 0, width, height), palette)
	fill(img, image.Rect(0, 0, width, height), bg)
	// title bar
	fill(img, image.Rect(0, 0, width, 34), barBg)
	circle(img, 20, 17, 6, red)
	circle(img, 40, 17, 6, ylw)
	circle(img, 60, 17, 6, grn2)
	drawText(img, face, 80, 22, strings.TrimSpace(*titleText), dim)

	y := topY
	drawRow := func(r row) {
		x := marginX
		for _, s := range r {
			drawText(img, face, x, y, s.text, s.col)
			x += textWidth(face, s.text)
		}
		y += lineH
	}
	for _, r := range rows {
		for _, vl := range wrapRow(r, maxChars) {
			drawRow(vl)
		}
	}
	if partial != nil {
		wrapped := wrapRow(*partial, maxChars)
		for i, vl := range wrapped {
			if i < len(wrapped)-1 {
				drawRow(vl)
				continue
			}
			// last visual line: draw it and place the cursor after it
			x := marginX
			for _, s := range vl {
				drawText(img, face, x, y, s.text, s.col)
				x += textWidth(face, s.text)
			}
			fill(img, image.Rect(x+1, y-11, x+9, y+3), green)
		}
	} else if len(rows) > 0 {
		fill(img, image.Rect(marginX+1, y-11, marginX+9, y+3), green)
	}
	return img
}

func fill(img *image.Paletted, r image.Rectangle, c color.RGBA) {
	idx := uint8(palette.Index(c))
	for yy := r.Min.Y; yy < r.Max.Y; yy++ {
		if yy < 0 || yy >= height {
			continue
		}
		for xx := r.Min.X; xx < r.Max.X; xx++ {
			if xx < 0 || xx >= width {
				continue
			}
			img.SetColorIndex(xx, yy, idx)
		}
	}
}

func circle(img *image.Paletted, cx, cy, r int, c color.RGBA) {
	for yy := -r; yy <= r; yy++ {
		for xx := -r; xx <= r; xx++ {
			if xx*xx+yy*yy <= r*r {
				img.SetColorIndex(cx+xx, cy+yy, uint8(palette.Index(c)))
			}
		}
	}
}

func drawText(img *image.Paletted, face font.Face, x, y int, s string, c color.RGBA) {
	d := &font.Drawer{
		Dst:  img,
		Src:  image.NewUniform(c),
		Face: face,
		Dot:  fixed.P(x, y),
	}
	d.DrawString(s)
}

func textWidth(face font.Face, s string) int {
	d := &font.Drawer{Face: face}
	return d.MeasureString(s).Ceil()
}
