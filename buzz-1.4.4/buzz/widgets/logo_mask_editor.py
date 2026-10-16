"""Interactive source-frame editor for immutable logo-mask snapshots."""
from __future__ import annotations

from dataclasses import replace
from pathlib import Path
import tempfile

from PyQt6.QtCore import QProcess, QRectF, Qt, QUrl, QSettings
from PyQt6.QtGui import QColor, QDesktopServices, QPen, QPixmap
from PyQt6.QtWidgets import (QButtonGroup, QCheckBox, QDialog, QDialogButtonBox, QDoubleSpinBox,
    QFileDialog, QFormLayout, QGraphicsItem, QGraphicsPixmapItem, QGraphicsRectItem,
    QGraphicsScene, QGraphicsView, QHBoxLayout, QLabel, QListWidget, QPushButton,
    QRadioButton, QSpinBox, QVBoxLayout, QWidget)
from buzz.localization.final_render import build_custom_logo_mask_filter_graph
from buzz.localization.logo_mask import LogoMaskConfig, LogoMaskRegion

_CORNERS = {"top-left": ("Trên trái", .015, .02), "top-right": ("Trên phải", .76, .02),
            "bottom-left": ("Dưới trái", .015, .85), "bottom-right": ("Dưới phải", .79, .85)}


class _MaskItem(QGraphicsRectItem):
    """Draggable normalized rectangle with four edges and four corner handles."""
    HANDLE = 8.0
    def __init__(self, region, image_rect):
        super().__init__(); self.region, self.image_rect, self._drag_kind, self.on_changed = region, image_rect, "", None
        self.setZValue(2); self.setPen(QPen(QColor("#ffcc33"), 2)); self.setBrush(QColor(255, 204, 51, 42))
        self.setFlag(QGraphicsItem.GraphicsItemFlag.ItemIsSelectable, True); self.setAcceptHoverEvents(True); self.set_region(region)
    def set_region(self, region):
        self.region = region; b = self.image_rect
        self.setRect(QRectF(b.x()+region.x*b.width(), b.y()+region.y*b.height(), region.w*b.width(), region.h*b.height())); self.setVisible(region.enabled)
    def _handle(self, p):
        r, h = self.rect(), self.HANDLE
        for key, x, y in (("nw",r.left(),r.top()),("ne",r.right(),r.top()),("sw",r.left(),r.bottom()),("se",r.right(),r.bottom())):
            if abs(p.x()-x)<=h and abs(p.y()-y)<=h: return key
        if abs(p.x()-r.left())<=h: return "w"
        if abs(p.x()-r.right())<=h: return "e"
        if abs(p.y()-r.top())<=h: return "n"
        if abs(p.y()-r.bottom())<=h: return "s"
        return "move" if r.contains(p) else ""
    def paint(self, painter, option, widget=None):
        super().paint(painter, option, widget); painter.setBrush(QColor("#ffcc33")); painter.setPen(Qt.PenStyle.NoPen); r, h = self.rect(), self.HANDLE/2
        for x,y in ((r.left(),r.top()),(r.center().x(),r.top()),(r.right(),r.top()),(r.left(),r.center().y()),(r.right(),r.center().y()),(r.left(),r.bottom()),(r.center().x(),r.bottom()),(r.right(),r.bottom())): painter.drawRect(QRectF(x-h,y-h,2*h,2*h))
    def mousePressEvent(self, event):
        self._drag_kind, self._start, self._start_rect = self._handle(event.pos()), event.pos(), QRectF(self.rect())
        if self._drag_kind: self.setSelected(True); event.accept()
        else: super().mousePressEvent(event)
    def mouseMoveEvent(self, event):
        if not self._drag_kind: return super().mouseMoveEvent(event)
        dx,dy = event.pos().x()-self._start.x(), event.pos().y()-self._start.y(); r,b,k = QRectF(self._start_rect),self.image_rect,self._drag_kind
        if k == "move": r.translate(dx,dy)
        else:
            if "w" in k: r.setLeft(r.left()+dx)
            if "e" in k: r.setRight(r.right()+dx)
            if "n" in k: r.setTop(r.top()+dy)
            if "s" in k: r.setBottom(r.bottom()+dy)
        if r.width()<2: r.setWidth(2)
        if r.height()<2: r.setHeight(2)
        if r.left()<b.left(): r.moveLeft(b.left())
        if r.top()<b.top(): r.moveTop(b.top())
        if r.right()>b.right(): r.moveRight(b.right())
        if r.bottom()>b.bottom(): r.moveBottom(b.bottom())
        self.setRect(r); event.accept()
    def mouseReleaseEvent(self, event):
        if not self._drag_kind: return super().mouseReleaseEvent(event)
        self._drag_kind=""; r,b=self.rect(),self.image_rect
        self.region=replace(self.region,x=(r.x()-b.x())/b.width(),y=(r.y()-b.y())/b.height(),w=r.width()/b.width(),h=r.height()/b.height())
        if self.on_changed: self.on_changed(self.region)
        event.accept()


class LogoMaskEditorDialog(QDialog):
    """Edits source masks only; preview never applies subtitles or Ben Review."""
    def __init__(self, config: LogoMaskConfig, parent=None, source_video: str | None = None):
        super().__init__(parent); self.setWindowTitle("Tùy chỉnh vùng che logo"); self.resize(980,700)
        self._regions, self._source = list(config.regions), source_video or ""
        self._frame_path = self._filtered_path = self._sample_path = None
        self._process=None; self._rendering_preview=False; self._updating=False; self._items={}; self._image_rect=QRectF(0,0,640,360)
        self.mode_widget = QWidget(self)
        self.legacy_radio,self.custom_radio=QRadioButton("Mặc định (như hiện tại)"),QRadioButton("Tùy chỉnh")
        # These controls have the same dialog parent, so Qt's implicit radio
        # grouping would otherwise make the preview selection clear the mode.
        self.mode_group = QButtonGroup(self)
        self.legacy_radio.setAutoExclusive(False)
        self.custom_radio.setAutoExclusive(False)
        self.mode_group.addButton(self.legacy_radio)
        self.mode_group.addButton(self.custom_radio)
        self.legacy_radio.setChecked(config.mode=="legacy"); self.custom_radio.setChecked(config.mode=="custom"); mode=QHBoxLayout(self.mode_widget); mode.addWidget(self.legacy_radio); mode.addWidget(self.custom_radio); mode.addStretch()
        self.source_label,pick=QLabel("Chưa chọn video"),QPushButton("Chọn video…"); pick.clicked.connect(self._pick_source); source_row=QHBoxLayout(); source_row.addWidget(self.source_label,1);source_row.addWidget(pick)
        self.scene,self.view=QGraphicsScene(self),QGraphicsView(); self.view.setScene(self.scene);self.view.setMinimumSize(500,350);self._pixmap_item=None
        self.corner_checks={}; corners=QHBoxLayout()
        for ident,(label,_,_) in _CORNERS.items():
            check=QCheckBox(label);check.setChecked(any(r.id==ident for r in self._regions));check.toggled.connect(lambda checked,i=ident:self._toggle_corner(i,checked));self.corner_checks[ident]=check;corners.addWidget(check)
        self.list=QListWidget();self.list.currentRowChanged.connect(self._load_region);add,remove=QPushButton("+ Thêm vùng tự do"),QPushButton("Xóa vùng");add.clicked.connect(self._add_region);remove.clicked.connect(self._remove_region);list_buttons=QHBoxLayout();list_buttons.addWidget(add);list_buttons.addWidget(remove)
        self.fields=[];form=QFormLayout()
        for label in ("x","y","w","h"):
            field=QDoubleSpinBox();field.setRange(0,100);field.setDecimals(3);field.setSuffix(" %");field.setSingleStep(1);field.valueChanged.connect(self._save_region);self.fields.append(field);form.addRow(label+" (%):",field)
        self.blur=QSpinBox();self.blur.setRange(1,40);self.blur.valueChanged.connect(self._save_region);form.addRow("Blur:",self.blur)
        self.enabled=QCheckBox("Bật vùng này");self.enabled.toggled.connect(self._save_region);form.addRow("",self.enabled)
        self.preview_widget = QWidget(self)
        self.original_button,self.filtered_button=QRadioButton("Xem gốc"),QRadioButton("Xem sau khi che")
        self.preview_group = QButtonGroup(self)
        self.original_button.setAutoExclusive(False)
        self.filtered_button.setAutoExclusive(False)
        self.preview_group.addButton(self.original_button)
        self.preview_group.addButton(self.filtered_button)
        self.original_button.setChecked(True);self.original_button.toggled.connect(lambda checked: checked and self._show_original_frame());self.filtered_button.toggled.connect(lambda checked: checked and self._render_filtered_frame());toggles=QHBoxLayout(self.preview_widget);toggles.addWidget(self.original_button);toggles.addWidget(self.filtered_button)
        self.sample_button,self.open_sample_button=QPushButton("Render thử 5 giây"),QPushButton("Mở video thử");self.sample_button.clicked.connect(self._render_sample);self.open_sample_button.clicked.connect(self._open_sample);self.open_sample_button.setEnabled(False);preview_buttons=QHBoxLayout();preview_buttons.addWidget(self.sample_button);preview_buttons.addWidget(self.open_sample_button)
        save_preset,load_preset=QPushButton("Lưu preset theo tỷ lệ"),QPushButton("Tải preset theo tỷ lệ");save_preset.clicked.connect(self._save_preset);load_preset.clicked.connect(self._load_preset);presets=QHBoxLayout();presets.addWidget(save_preset);presets.addWidget(load_preset)
        self.status=QLabel();self.status.setWordWrap(True);footer=QDialogButtonBox(QDialogButtonBox.StandardButton.Ok|QDialogButtonBox.StandardButton.Cancel);footer.accepted.connect(self.accept);footer.rejected.connect(self.reject)
        right=QVBoxLayout();right.addLayout(corners);right.addWidget(QLabel("Các vùng che"));right.addWidget(self.list);right.addLayout(list_buttons);right.addLayout(form);right.addWidget(self.preview_widget);right.addLayout(preview_buttons);right.addLayout(presets);right.addWidget(self.status);right.addWidget(footer)
        body=QHBoxLayout();body.addWidget(self.view,3);body.addLayout(right,2);layout=QVBoxLayout(self);layout.addWidget(self.mode_widget);layout.addLayout(source_row);layout.addLayout(body);self._refresh()
        if self._source:self._load_source(self._source)
    @property
    def config(self): return LogoMaskConfig(mode="custom" if self.custom_radio.isChecked() else "legacy",regions=tuple(self._regions))
    def _pick_source(self):
        source,_=QFileDialog.getOpenFileName(self,"Chọn video xem trước",self._source,"Video (*.mp4 *.mkv *.mov *.avi *.webm *.m4v);;All Files (*)")
        if source:self._load_source(source)
    def _preset_key(self):
        # The displayed, rotation-normalized frame is the authority here.
        orientation="portrait" if self._image_rect.height()>self._image_rect.width() else "landscape"
        return f"localization/logo-mask-preset/{orientation}"
    def _save_preset(self):
        QSettings().setValue(self._preset_key(),self.config.to_json());self.status.setText("Đã lưu preset riêng cho tỷ lệ video này.")
    def _load_preset(self):
        value=QSettings().value(self._preset_key(),"")
        try: config=LogoMaskConfig.from_json(str(value))
        except ValueError: self.status.setText("Chưa có preset hợp lệ cho tỷ lệ này.");return
        self._regions=list(config.regions);self.custom_radio.setChecked(config.mode=="custom");self.legacy_radio.setChecked(config.mode=="legacy")
        for ident,check in self.corner_checks.items():check.blockSignals(True);check.setChecked(any(r.id==ident for r in self._regions));check.blockSignals(False)
        self._refresh();self.status.setText("Đã tải preset theo tỷ lệ video.")
    def _load_source(self,source):
        if not Path(source).is_file():self.status.setText("Không tìm thấy video đã chọn.");return
        self._source=source;self.source_label.setText(Path(source).name);self.status.setText("Đang trích frame video…");self._frame_path=Path(tempfile.gettempdir())/f"buzz-logo-frame-{id(self)}.png";self._run(["-ss","0","-i",source,"-frames:v","1","-vf","format=rgb24",str(self._frame_path)],self._frame_ready)
    def _run(self,args,callback):
        if self._process is not None:self._process.kill()
        p=QProcess(self);p.setProgram("ffmpeg");p.setArguments(["-y","-nostdin",*args]);p.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels);p.finished.connect(lambda code,state,cb=callback,proc=p:cb(code,bytes(proc.readAllStandardOutput()).decode("utf-8","replace")));self._process=p;p.start()
    def _frame_ready(self,code,output):
        if code or not self._frame_path or not self._frame_path.is_file():self.status.setText("Không thể trích frame: "+output[-300:]);return
        self._show_pixmap(QPixmap(str(self._frame_path)));self.status.setText("Kéo vùng hoặc 8 nút vuông để chỉnh kích thước.")
    def _show_pixmap(self,pixmap):
        if pixmap.isNull():return
        self.scene.clear();self._items={};self._image_rect=QRectF(0,0,pixmap.width(),pixmap.height());self._pixmap_item=QGraphicsPixmapItem(pixmap);self.scene.addItem(self._pixmap_item);self.scene.setSceneRect(self._image_rect);self._rebuild_items();self.view.fitInView(self._image_rect,Qt.AspectRatioMode.KeepAspectRatio)
    def resizeEvent(self,event):
        super().resizeEvent(event)
        if self._pixmap_item:self.view.fitInView(self._image_rect,Qt.AspectRatioMode.KeepAspectRatio)
    def _rebuild_items(self):
        if not self._pixmap_item:return
        for item in self._items.values():self.scene.removeItem(item)
        self._items={}
        for region in self._regions:
            item=_MaskItem(region,self._image_rect);item.on_changed=self._item_changed;self.scene.addItem(item);self._items[region.id]=item
    def _refresh(self):
        row=self.list.currentRow();self.list.blockSignals(True);self.list.clear()
        for region in self._regions:self.list.addItem(region.name)
        self.list.blockSignals(False)
        if self._regions:self.list.setCurrentRow(max(0,min(row,len(self._regions)-1)))
        self._rebuild_items()
    def _toggle_corner(self,ident,checked):
        existing=next((r for r in self._regions if r.id==ident),None)
        if checked and not existing:name,x,y=_CORNERS[ident];self._regions.append(LogoMaskRegion(ident,name,True,x,y,.20,.10,24))
        elif not checked and existing:self._regions.remove(existing)
        self._refresh();self._refresh_filtered_preview()
    def _add_region(self):
        number=1;ids={r.id for r in self._regions}
        while f"zone-{number}" in ids:number+=1
        self._regions.append(LogoMaskRegion(f"zone-{number}",f"Vùng {number}",True,.4,.4,.2,.1,24));self._refresh();self.list.setCurrentRow(len(self._regions)-1);self._refresh_filtered_preview()
    def _remove_region(self):
        row=self.list.currentRow()
        if row<0:return
        ident=self._regions[row].id;self._regions.pop(row)
        if ident in self.corner_checks:self.corner_checks[ident].blockSignals(True);self.corner_checks[ident].setChecked(False);self.corner_checks[ident].blockSignals(False)
        self._refresh();self._refresh_filtered_preview()
    def _load_region(self,row):
        if row<0 or row>=len(self._regions):return
        self._updating=True;region=self._regions[row]
        for field,value in zip(self.fields,(region.x,region.y,region.w,region.h)):field.setValue(value*100)
        self.blur.setValue(region.blur_radius);self.enabled.setChecked(region.enabled);self._updating=False
        if region.id in self._items:self._items[region.id].setSelected(True)
    def _save_region(self):
        if self._updating:return
        row=self.list.currentRow()
        if row<0:return
        old=self._regions[row];x,y,w,h=(field.value()/100 for field in self.fields);x,y=min(x,.9999),min(y,.9999);w,h=min(max(w,.0001),1-x),min(max(h,.0001),1-y);self._regions[row]=replace(old,enabled=self.enabled.isChecked(),x=x,y=y,w=w,h=h,blur_radius=self.blur.value())
        if old.id in self._items:self._items[old.id].set_region(self._regions[row])
        self._refresh_filtered_preview()
    def _item_changed(self,region):
        row=next((i for i,old in enumerate(self._regions) if old.id==region.id),-1)
        if row>=0:self._regions[row]=region;self.list.setCurrentRow(row);self._load_region(row);self._refresh_filtered_preview()
    def _refresh_filtered_preview(self):
        if self.filtered_button.isChecked() and self._source:
            self._render_filtered_frame()
    def _render_filtered_frame(self):
        if not self._source:self.status.setText("Chọn video trước khi xem frame đã che.");self.original_button.setChecked(True);return
        self._filtered_path=Path(tempfile.gettempdir())/f"buzz-logo-filtered-{id(self)}.png";graph=build_custom_logo_mask_filter_graph(LogoMaskConfig(mode="custom",regions=tuple(self._regions)));self.status.setText("Đang tạo frame đã che…");self._run(["-ss","0","-i",self._source,"-filter_complex",graph,"-map","[vout]","-frames:v","1",str(self._filtered_path)],self._filtered_ready)
    def _filtered_ready(self,code,output):
        if code or not self._filtered_path or not self._filtered_path.is_file():self.status.setText("Không thể tạo preview: "+output[-300:]);return
        self._show_pixmap(QPixmap(str(self._filtered_path)));self.status.setText("Đang xem frame đã che thực tế.")
    def _show_original_frame(self):
        if self._frame_path and self._frame_path.is_file():self._show_pixmap(QPixmap(str(self._frame_path)));self.status.setText("Đang xem frame gốc.")
    def _render_sample(self):
        if self._rendering_preview:
            if self._process:self._process.kill()
            if self._sample_path:
                try: self._sample_path.unlink(missing_ok=True)
                except OSError: pass
            self._rendering_preview=False;self.sample_button.setText("Render thử 5 giây");self.status.setText("Đã hủy render thử.");return
        if not self._source:self.status.setText("Chọn video trước khi render thử.");return
        self._sample_path=Path(tempfile.gettempdir())/f"buzz-logo-preview-{id(self)}.mp4";graph=build_custom_logo_mask_filter_graph(LogoMaskConfig(mode="custom",regions=tuple(self._regions)));self._rendering_preview=True;self.sample_button.setText("Hủy preview");self.status.setText("Đang render 5 giây…");self._run(["-ss","0","-i",self._source,"-t","5","-filter_complex",graph,"-map","[vout]","-map","0:a?","-c:v","libx264","-c:a","aac","-movflags","+faststart",str(self._sample_path)],self._sample_ready)
    def _sample_ready(self,code,output):
        self._rendering_preview=False;self.sample_button.setText("Render thử 5 giây")
        if code or not self._sample_path or not self._sample_path.is_file():self.status.setText("Preview không thành công: "+output[-300:]);return
        self.open_sample_button.setEnabled(True);self.status.setText("Preview 5 giây đã sẵn sàng.")
    def _open_sample(self):
        if self._sample_path and self._sample_path.is_file():QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._sample_path)))
    def closeEvent(self,event):
        if self._process is not None:
            self._process.kill()
        for path in (self._frame_path,self._filtered_path,self._sample_path):
            if path:
                try: path.unlink(missing_ok=True)
                except OSError: pass
        super().closeEvent(event)
