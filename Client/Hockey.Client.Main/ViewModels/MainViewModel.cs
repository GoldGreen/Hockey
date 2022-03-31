using Hockey.Client.Main.Abstractions;
using Microsoft.Win32;
using OpenCvSharp;
using OpenCvSharp.WpfExtensions;
using ReactiveUI;
using ReactiveUI.Fody.Helpers;
using System;
using System.Linq;
using System.Reactive;
using System.Reactive.Linq;
using System.Reactive.Subjects;
using System.Threading;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace Hockey.Client.Main.ViewModels
{
    internal class MainViewModel : ReactiveObject
    {
        public IMainModel Model { get; }

        [Reactive] private VideoCapture VideoCapture { get; set; }
        [Reactive] public bool IsPaused { get; set; } = true;

        [Reactive] public int FrameNum { get; set; }
        [Reactive] public bool FrameToMinimapFrame { get; set; } = true;
        [Reactive] public int MinimapFrameNum { get; set; }

        [Reactive] public int FramesCount { get; set; }
        [Reactive] public string FilePath { get; set; }

        [Reactive] public string FirstTeamName { get; set; }
        [Reactive] public Color FirstTeamColor { get; set; } = Color.FromRgb(204, 204, 204);
        public ICommand PickFirstColorCommand { get; }
        [Reactive] public bool FirstColorPicked { get; set; }

        [Reactive] public string SecondTeamName { get; set; }
        [Reactive] public Color SecondTeamColor { get; set; } = Color.FromRgb(93, 155, 155);
        [Reactive] public bool SecondColorPicked { get; set; }

        [Reactive] public ImageSource FrameSource { get; set; }
        [Reactive] public ImageSource MinimapSource { get; set; }

        [Reactive] private Mat Frame { get; set; }
        public ICommand PickSecondtColorCommand { get; }

        public ICommand OpenVideoCommand { get; }
        public ICommand StartDetectionCommand { get; }
        public ICommand LoadFramesInfoCommand { get; }

        public ICommand ReversePausedCommand { get; }
        public ICommand CloseVideoCommand { get; }

        public ICommand PickColorCommand { get; }


        private readonly Subject<Unit> _onReadingCrash = new();

        private readonly Mat _minimap = new("Resources/minimap.png");

        public MainViewModel(IMainModel model)
        {
            Model = model;

            OpenVideoCommand = ReactiveCommand.Create
            (
                OpenVideo,
                this.WhenAnyValue(x => x.VideoCapture, x => x.IsPaused, (cap, isP) => cap == null && isP)
            );

            PickFirstColorCommand = ReactiveCommand.Create
            (
                () => { FirstColorPicked = true; },
                this.WhenAnyValue(x => x.VideoCapture,
                                  x => x.IsPaused,
                                  x => x.FirstColorPicked,
                                  x => x.SecondColorPicked,
                                  (cap, isP, pf, ps) => cap != null && isP && !pf && !ps)
            );

            PickSecondtColorCommand = ReactiveCommand.Create
            (
                () => { SecondColorPicked = true; },
                this.WhenAnyValue(x => x.VideoCapture,
                                  x => x.IsPaused,
                                  x => x.FirstColorPicked,
                                  x => x.SecondColorPicked,
                                  (cap, isP, pf, ps) => cap != null && isP && !pf && !ps)
            );

            PickColorCommand = ReactiveCommand.Create<MouseButtonEventArgs>
            (
                x =>
                {
                    var image = x.Source as Image;

                    var pos = x.GetPosition(image);
                    (byte b, byte g, byte r) = Frame.At<Vec3b>
                    (
                        (int)(pos.Y / image.ActualHeight * Frame.Height),
                        (int)(pos.X / image.ActualWidth * Frame.Width)
                    );

                    var color = Color.FromRgb(r, g, b);

                    if (FirstColorPicked)
                    {
                        FirstColorPicked = false;
                        FirstTeamColor = color;
                    }
                    else if (SecondColorPicked)
                    {
                        SecondColorPicked = false;
                        SecondTeamColor = color;
                    }
                },
                this.WhenAnyValue(x => x.FirstColorPicked,
                                  x => x.SecondColorPicked,
                                  x => x.Frame,
                                  (pf, ps, fr) => (pf || ps) && fr != null)
            );

            ReversePausedCommand = ReactiveCommand.Create
            (
                () => IsPaused = !IsPaused,
                this.WhenAnyValue(x => x.VideoCapture).Select(cap => cap != null)
            );

            StartDetectionCommand = ReactiveCommand.CreateFromTask
            (
                () => Model.StartDetectingVideo
                (
                    new()
                    {
                        FileName = FilePath,
                        FirstTeamName = FirstTeamName,
                        FirstTeamColor = new byte[]
                        {
                            FirstTeamColor.R,
                            FirstTeamColor.B,
                            FirstTeamColor.G
                        },
                        SecondTeamName = SecondTeamName,
                        SecondTeamColor = new byte[]
                        {
                            SecondTeamColor.R,
                            SecondTeamColor.G,
                            SecondTeamColor.B
                        }
                    }
                ),
                this.WhenAnyValue(x => x.FilePath,
                                  x => x.FirstTeamName,
                                  x => x.SecondTeamName,
                                  (f, fn, sn) => new[] { f, fn, sn }.All(x => !string.IsNullOrWhiteSpace(x)))
            );

            LoadFramesInfoCommand = ReactiveCommand.CreateFromTask
            (
                async () => Model.FramesInfo = await Model.GetAllFrames()
            );

            this.WhenAnyValue(x => x.FrameNum)
                .Where(_ => IsPaused && VideoCapture != null)
                .Throttle(TimeSpan.FromMilliseconds(100))
                .ObserveOnDispatcher()
                .Subscribe
                (
                    num =>
                    {
                        Frame = new();
                        VideoCapture.Set(VideoCaptureProperties.PosFrames, num);
                        VideoCapture.Read(Frame);
                        DrawPeople(Frame);
                        Application.Current.Dispatcher.Invoke
                        (
                            () => FrameSource = Frame.ToBitmapSource()
                        );
                    }
                );

            this.WhenAnyValue(x => x.FrameNum, x => x.FrameToMinimapFrame, (num, use) => (num, use))
                .Where(x => x.use)
                .Select(x => x.num)
                .ObserveOnDispatcher()
                .Subscribe(x => MinimapFrameNum = x);

            this.WhenAnyValue(x => x.MinimapFrameNum)
                .Subscribe(_ => DrawMinimap());

            _onReadingCrash.ObserveOnDispatcher()
                           .Subscribe(_ =>
                           {
                               if (VideoCapture?.IsDisposed ?? false)
                               {
                                   VideoCapture.Dispose();
                                   VideoCapture = null;
                               }
                           });
        }


        private void DrawPeople(Mat frame)
        {
            if (Model.FramesInfo != null && Model.FramesInfo.TryGetValue(FrameNum, out var players))
            {
                foreach (var player in players)
                {
                    int[] bbox = player.Bbox.Select(x => (int)x).ToArray();

                    if (!string.IsNullOrEmpty(player.Team))
                    {
                        Cv2.PutText(frame,
                                    player.Team,
                                    new(bbox[0], bbox[1] - 10),
                                    HersheyFonts.HersheySimplex,
                                    0.75,
                                    new(50, 10, 24),
                                    2);
                    }

                    Color color = player.Team == FirstTeamName ? FirstTeamColor : player.Team == SecondTeamName ? SecondTeamColor : Brushes.Black.Color;

                    Cv2.Rectangle(frame, new(bbox[0], bbox[1]), new(bbox[2], bbox[3]), new(0, 0, 0), 3);
                    Cv2.Rectangle(frame, new(bbox[0], bbox[1]), new(bbox[2], bbox[3]), new(color.B, color.G, color.R), 2);
                }
            }
        }

        public void DrawMinimap()
        {
            using var minimap = _minimap.Clone();

            double width = minimap.Width * 0.9;
            double height = minimap.Height * 0.8;

            if (Model.FramesInfo != null && Model.FramesInfo.TryGetValue(MinimapFrameNum, out var players))
            {
                foreach (var player in players)
                {
                    Color? color = player.Team == FirstTeamName ? FirstTeamColor : player.Team == SecondTeamName ? SecondTeamColor : null;

                    Cv2.Ellipse(minimap,
                               new(player.FieldCoordinate[0] * width, player.FieldCoordinate[1] * height),
                               new(60, 60),
                               0,
                               0,
                               360,
                               new(0, 0, 0),
                               5);

                    if (color != null)
                    {
                        Cv2.Ellipse(minimap,
                                    new(player.FieldCoordinate[0] * width, player.FieldCoordinate[1] * height),
                                    new(50, 50),
                                    0,
                                    0,
                                    360,
                                    new(color.Value.B, color.Value.G, color.Value.R),
                                    -1);
                    }
                }
            }

            MinimapSource = minimap.ToBitmapSource();
        }

        private void OpenVideo()
        {
            OpenFileDialog openFileDialog = new();

            if (openFileDialog.ShowDialog() == true)
            {
                FilePath = openFileDialog.FileName;
                StartVideo(FilePath);
            }
        }

        private Task StartVideo(string name)
        {
            IsPaused = true;

            VideoCapture?.Dispose();
            VideoCapture = new VideoCapture(name);

            FrameNum = 0;
            FramesCount = VideoCapture.FrameCount;

            return Task.Run
            (
                () =>
                {
                    try
                    {
                        while (true)
                        {
                            if (IsPaused)
                            {
                                Thread.Sleep(500);
                                continue;
                            }

                            Frame = new Mat();

                            FrameNum = VideoCapture.PosFrames;

                            if (!VideoCapture.Read(Frame))
                            {
                                break;
                            }

                            DrawPeople(Frame);

                            Application.Current.Dispatcher.Invoke
                            (
                                () => FrameSource = Frame.ToBitmapSource()
                            );

                            Thread.Sleep((int)(1000 / VideoCapture.Fps));
                        }
                    }
                    catch (Exception e)
                    {
                        MessageBox.Show(e.Message);
                        _onReadingCrash.OnNext(Unit.Default);
                    }
                    finally
                    {
                        IsPaused = true;
                    }
                }
            );
        }
    }
}
